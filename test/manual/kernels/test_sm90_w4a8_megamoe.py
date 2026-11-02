"""Hopper integration smoke: packed loading, CP views, empty rank, graph replay.

Run with torchrun --standalone --nproc-per-node=4. This is a small routed-MoE
fixture, not a full-model CP/BCG accuracy or performance acceptance test.
The FP32 reference uses the same FP4 checkpoint weights; the 5% relative-L2
bound allows the fused path's two FP8 activation quantizations.
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F


def unpack(weight, scale):
    raw = weight.view(torch.uint8)
    nibble = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weight.device)
    values = lut[(nibble & 7).long()] * torch.where((nibble & 8) != 0, -1, 1)
    return values * scale.repeat_interleave(32, dim=-1)


def reference(x, ids, weights, w13, w2):
    y = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(w13.shape[0]):
        rows, slots = torch.where(ids == expert)
        if rows.numel():
            gate, up = F.linear(x[rows].float(), w13[expert]).chunk(2, dim=-1)
            values = F.linear(F.silu(gate) * up, w2[expert])
            y.index_add_(0, rows, values * weights[rows, slots, None] * 1.5)
    return y


@torch.inference_mode()
def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This smoke requires SM90 GPUs and a DeepGEMM PR #53 build")
    dist.init_process_group("nccl")
    import deep_gemm

    from sglang.srt.layers.moe.mega_moe_sm90 import (
        build_sm90_fp4_mega_moe_experts_weights,
        run_sm90_mega_routed,
    )

    world = dist.get_world_size()
    # The symmetric workspace's FP32 g64 scale rows must be 16-byte aligned.
    hidden, intermediate, local_experts, topk, capacity = 512, 512, 2, 2, 16
    torch.manual_seed(6318 + rank)
    pairs, refs = [], []
    for n, k in ((2 * intermediate, hidden), (hidden, intermediate)):
        weight = torch.randint(
            -128, 128, (local_experts, n, k // 2), device="cuda", dtype=torch.int8
        )
        scale = torch.full((local_experts, n, k // 32), 2**-7, device="cuda")
        pairs.append((weight, scale))
        ref = unpack(weight, scale)
        all_refs = [torch.empty_like(ref) for _ in range(world)]
        dist.all_gather(all_refs, ref)
        refs.append(torch.cat(all_refs))
    experts = SimpleNamespace(
        w13_weight=torch.nn.Parameter(pairs[0][0], requires_grad=False),
        w13_weight_scale_inv=torch.nn.Parameter(pairs[0][1], requires_grad=False),
        w2_weight=torch.nn.Parameter(pairs[1][0], requires_grad=False),
        w2_weight_scale_inv=torch.nn.Parameter(pairs[1][1], requires_grad=False),
        _mega_moe_w4a4=False,
    )
    build_sm90_fp4_mega_moe_experts_weights(experts)
    assert experts.w13_weight.dtype == torch.int8
    del pairs
    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD,
        world * local_experts,
        capacity,
        topk,
        hidden,
        intermediate,
        mma_type="fp8xfp4",
        activation="swiglu",
    )

    def check(y, x, ids, weights):
        expected = reference(x, ids, weights, *refs)
        error = float((y.float() - expected).norm() / expected.norm().clamp_min(1e-30))
        assert bool(torch.isfinite(y).all()) and error < 0.05, (rank, error)
        padding = (ids < 0).all(dim=1)
        assert not bool(y[padding].ne(0).any()), (
            "padding must not dispatch stale tokens"
        )
        return error

    def run(x, ids, weights):
        return run_sm90_mega_routed(
            experts,
            x,
            ids,
            weights,
            buf,
            x.shape[0],
            hidden_size=hidden,
            routed_scaling_factor=1.5,
        )

    try:
        # Identical global input, interleaved into non-contiguous CP-local views.
        torch.manual_seed(100)
        global_x = (
            torch.randn(world * 3 + 1, hidden, device="cuda", dtype=torch.bfloat16)
            * 0.1
        )
        for case in ("cp_view", "empty_rank", "graph_bucket"):
            x = global_x[rank::world]
            if case == "empty_rank" and rank == 0:
                x = x[:0]
            valid = x.shape[0]
            if case == "graph_bucket":
                padded = x.new_zeros((capacity, hidden))
                padded[:valid].copy_(x)
                x = padded
            ids = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device="cuda")
            ids[:valid] = torch.arange(valid * topk, device="cuda").view(
                valid, topk
            ) % (world * local_experts)
            weights = torch.zeros_like(ids, dtype=torch.float32)
            weights[:valid].fill_(1 / topk)
            for _ in range(3):
                y = run(x, ids, weights)
            torch.cuda.synchronize()
            error = check(y, x, ids, weights)
            if case == "graph_bucket":
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    y = run(x, ids, weights)
                for _ in range(3):
                    x[:valid].mul_(0.75)
                    ids[:valid].add_(1).remainder_(world * local_experts)
                    graph.replay()
                    torch.cuda.synchronize()
                    error = check(y, x, ids, weights)
            print(
                f"rank={rank} case={case} rows={x.shape[0]} "
                f"valid={valid} relative_l2={error:.6f}",
                flush=True,
            )
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
