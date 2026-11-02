"""Four-GPU H20 routed-MoE chain comparison, including communication.

Run with torchrun --standalone --nproc-per-node=4. Same checkpoint-format
weights/routes for Humming EP4, Humming TP4/EP1 and packed MegaMoE EP4.
Includes quantization, routing preparation, both expert GEMMs, activation,
weighted combination and input/output collectives. Router and shared expert
are measured separately in the real server trace, not emulated here.
"""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

SCALE_JITTER = 0


def raw_weights(experts, hidden, intermediate, seed, part=None):
    """Stable per-expert tensors let TP and EP consume identical full weights."""
    result = []
    for stage, n, k in ((0, 2 * intermediate, hidden), (1, hidden, intermediate)):
        weights, scales = [], []
        for expert in experts:
            gen = torch.Generator(device="cuda").manual_seed(seed + expert * 2 + stage)
            weight = torch.randint(
                0, 256, (n, k // 2), generator=gen, dtype=torch.uint8, device="cuda"
            )
            scale = torch.full((n, k // 32), 120, device="cuda", dtype=torch.uint8)
            if SCALE_JITTER:
                scale = torch.randint(
                    120 - SCALE_JITTER,
                    121 + SCALE_JITTER,
                    (n, k // 32),
                    generator=gen,
                    dtype=torch.uint8,
                    device="cuda",
                )
            if part is not None:
                rank, world = part
                width = intermediate // world
                start, end = rank * width, (rank + 1) * width
                if stage == 0:
                    weight = torch.cat(
                        (
                            weight[start:end],
                            weight[intermediate + start : intermediate + end],
                        )
                    )
                    scale = torch.cat(
                        (
                            scale[start:end],
                            scale[intermediate + start : intermediate + end],
                        )
                    )
                else:
                    weight = weight[:, start // 2 : end // 2].contiguous()
                    scale = scale[:, start // 32 : end // 32].contiguous()
            weights.append(weight)
            scales.append(scale)
        result.append((torch.stack(weights), torch.stack(scales)))
    return result


def humming_runner(raw, hidden, intermediate, experts, world, topk):
    from humming.layer import HummingMethod
    from humming.schema import BaseWeightSchema, HummingInputSchema

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore

    layer = torch.nn.Module()
    layer.hidden_size = hidden
    layer.intermediate_size_per_partition = intermediate
    layer.params_dtype = torch.bfloat16
    local_experts = raw[0][0].shape[0]
    schema = BaseWeightSchema.from_config({"quant_method": "mxfp4"})
    for name, pair, n, k in zip(
        ("w13", "w2"), raw, (2 * intermediate, hidden), (hidden, intermediate)
    ):
        ws, tensors = schema.convert_humming(
            tensors={"weight": pair[0].clone(), "weight_scale": pair[1].clone()},
            shape_n_stacks=[n // 2, n // 2] if name == "w13" else [n],
            shape_k_stacks=[k],
            param_dtype=torch.bfloat16,
            num_experts=local_experts,
        )
        for key, value in tensors.items():
            layer.register_parameter(
                name + "_" + key, torch.nn.Parameter(value, requires_grad=False)
            )
        HummingMethod.prepare_layer_meta(
            layer,
            shape_n=n,
            shape_k=k,
            weight_schema=ws,
            input_schema=HummingInputSchema.from_config({"dtype": "float8e4m3"}),
            num_experts=local_experts,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            has_bias=False,
            torch_dtype=torch.bfloat16,
            sublayer_name=name,
        )
        HummingMethod.transform_humming_layer(layer, sublayer_name=name)
    layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device="cuda"))
    runner = HummingRunnerCore(
        MoeRunnerConfig(
            num_experts=experts,
            num_local_experts=local_experts,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            top_k=topk,
            params_dtype=torch.bfloat16,
            activation="silu",
            is_gated=True,
            swiglu_limit=10.0,
            routed_scaling_factor=1.0,
            gate_up_interleaved=False,
        )
    )
    runner.layer = layer
    return runner


def summary(values):
    return dict(
        median_us=statistics.median(values),
        samples_us=values,
        cv=statistics.pstdev(values) / statistics.mean(values),
    )


def reference_rows(x, ids, weights, experts, hidden, intermediate, rank, world, topk):
    """FP32 reference for a small subset; no FP8 activation quantization."""
    import torch.nn.functional as F

    count = min(x.shape[0], 16)
    allx = x.new_empty((count * world, hidden))
    allids = ids.new_empty((count * world, topk))
    allweights = weights.new_empty((count * world, topk))
    dist.all_gather_into_tensor(allx, x[:count].contiguous())
    dist.all_gather_into_tensor(allids, ids[:count].contiguous())
    dist.all_gather_into_tensor(allweights, weights[:count].contiguous())
    output = torch.zeros(count * world, hidden, device="cuda", dtype=torch.float32)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")

    def unpack(pair):
        raw, scales = pair[0][0], pair[1][0]
        codes = torch.stack((raw & 15, raw >> 4), -1).flatten(-2)
        value = lut[(codes & 7).long()] * torch.where((codes & 8) != 0, -1, 1)
        return value * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)

    for expert in range(rank * (experts // world), (rank + 1) * (experts // world)):
        rows, slots = torch.where(allids == expert)
        if not rows.numel():
            continue
        raw = raw_weights([expert], hidden, intermediate, 6318)
        gate, up = F.linear(allx[rows].float(), unpack(raw[0])).chunk(2, -1)
        value = F.linear(F.silu(gate.clamp(max=10)) * up.clamp(-10, 10), unpack(raw[1]))
        output.index_add_(0, rows, value * allweights[rows, slots, None])
    dist.all_reduce(output)
    return output[rank * count : (rank + 1) * count]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline-stages", nargs="+", type=int, default=[])
    parser.add_argument("--rows", nargs="+", type=int, default=[32, 96, 1024, 4096])
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--sms", nargs="+", type=int, default=[78, 72, 64])
    parser.add_argument(
        "--capacity",
        type=int,
        default=128,
        help="Minimum symmetric-buffer token capacity per rank",
    )
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--tp-baseline", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--scale-jitter", type=int, default=0)
    parser.add_argument("--fragmentpack", action="store_true")
    parser.add_argument("--pairpack", action="store_true")
    parser.add_argument("--signpack", action="store_true")
    parser.add_argument(
        "--native-register-pack",
        action="store_true",
        help="Use the native DeepGEMM SM90 register-weight API",
    )
    parser.add_argument("--dump-output", action="store_true")
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--ncu-variant", choices=("mega", "humming"), default="mega")
    args = parser.parse_args()
    global SCALE_JITTER
    SCALE_JITTER = args.scale_jitter
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    import deep_gemm

    world = dist.get_world_size()
    hidden, intermediate = 5120, 2304
    local_experts = args.experts // world
    args.output.mkdir(parents=True, exist_ok=True)
    raw = raw_weights(
        range(rank * local_experts, (rank + 1) * local_experts),
        hidden,
        intermediate,
        6318,
    )
    ep_runner = humming_runner(
        raw, hidden, intermediate, args.experts, world, args.topk
    )
    prepared = deep_gemm.transform_weights_for_mega_moe_sm90_fp4(
        (raw[0][0].view(torch.int8), raw[0][1].view(torch.float8_e8m0fnu).float()),
        (raw[1][0].view(torch.int8), raw[1][1].view(torch.float8_e8m0fnu).float()),
    )
    if args.native_register_pack:
        if args.fragmentpack or args.pairpack or args.signpack:
            parser.error(
                "Native register packing already applies all three permutations"
            )
        prepared = tuple(
            (deep_gemm.pack_sm90_fp4_register_weights(w), sf) for w, sf in prepared
        )
    if args.fragmentpack:

        def pack_fragments(w):
            shape = w.shape
            index = torch.tensor(
                [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15], device=w.device
            )
            return w.reshape(*shape[:-1], -1, 16).index_select(-1, index).reshape_as(w)

        prepared = tuple((pack_fragments(w), sf) for w, sf in prepared)
    if args.pairpack:

        def pack_pairs(w):
            e, n, kbytes = w.shape
            return (
                w.reshape(e, n // 2, 2, kbytes // 64, 64)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .reshape_as(w)
            )

        prepared = tuple((pack_pairs(w), sf) for w, sf in prepared)
    if args.signpack:

        def repack_signs(weight):
            result = torch.empty_like(weight)
            for e in range(weight.shape[0]):
                src = weight[e].view(torch.int32)
                dst = src & 0x77777777
                for dest, source in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
                    dst |= ((src >> (source * 4)) & 8) << (dest * 4)
                result[e].copy_(dst.view(torch.int8).reshape_as(weight[e]))
            return result

        prepared = tuple((repack_signs(w), sf) for w, sf in prepared)
    del raw
    gc.collect()
    torch.cuda.empty_cache()
    tp_runner = None
    if args.tp_baseline:
        raw = raw_weights(
            range(args.experts), hidden, intermediate, 6318, part=(rank, world)
        )
        tp_runner = humming_runner(
            raw, hidden, intermediate // world, args.experts, world, args.topk
        )
        del raw
        gc.collect()
        torch.cuda.empty_cache()
    records = []
    for rows in args.rows:
        torch.manual_seed(9000 + rank + rows)
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
        ids = (
            torch.rand(rows, args.experts, device="cuda").topk(args.topk).indices.int()
        )
        weights = torch.rand(rows, args.topk, device="cuda")
        weights.div_(weights.sum(-1, keepdim=True))
        allx = x.new_empty((rows * world, hidden))
        allids = ids.new_empty((rows * world, args.topk))
        allweights = weights.new_empty((rows * world, args.topk))
        dist.all_gather_into_tensor(allids, ids)
        dist.all_gather_into_tensor(allweights, weights)
        epids = allids - rank * local_experts
        epids = torch.where((epids >= 0) & (epids < local_experts), epids, -1)
        result = x.new_empty(x.shape)
        buffers = {}
        for sms in args.sms:
            deep_gemm.set_num_sms(sms)
            buffers[sms] = deep_gemm.get_symm_buffer_for_mega_moe(
                dist.group.WORLD,
                args.experts,
                max(rows, args.capacity),
                args.topk,
                hidden,
                intermediate,
                mma_type="fp8xfp4",
            )

        def humming(runner, routes):
            dist.all_gather_into_tensor(allx, x)
            partial = runner._run_indexed_gemm(allx, allweights, routes)
            dist.reduce_scatter_tensor(result, partial)
            return result

        def mega(sms, stages=None):
            if stages is not None:
                os.environ["DG_SM90_FP4_REGISTER_STAGES"] = str(stages)
            deep_gemm.set_num_sms(sms)
            buf = buffers[sms]
            deep_gemm.mega_moe_pre_dispatch_sm90(
                x,
                ids,
                weights,
                buf.x,
                buf.x_sf,
                buf.topk_idx,
                buf.topk_weights,
                num_tokens=rows,
                group_size=128,
            )
            deep_gemm.fp8_fp4_mega_moe(
                result, *prepared, buf, recipe=(1, 1, 32), activation_clamp=10.0
            )
            return result

        functions = {"humming_ep4": lambda: humming(ep_runner, epids)}
        if tp_runner is not None:
            functions["humming_tp4_ep1"] = lambda: humming(tp_runner, allids)
        for sms in args.sms:
            if args.pipeline_stages:
                assert not args.ncu
                for stages in args.pipeline_stages:
                    functions[f"mega_sms{sms}_stages{stages}"] = (
                        lambda sms=sms, stages=stages: mega(sms, stages)
                    )
            else:
                functions[f"mega_sms{sms}"] = lambda sms=sms: mega(sms)
        graphs, errors = {}, {}
        expected = functions["humming_ep4"]().clone()
        reference = reference_rows(
            x, ids, weights, args.experts, hidden, intermediate, rank, world, args.topk
        )
        reference_errors = {}
        for name, fn in functions.items():
            if name.startswith("mega"):
                deep_gemm.set_num_sms(int(name.split("sms")[1].split("_")[0]))
            for _ in range(3):
                actual = fn()
            torch.cuda.synchronize()
            err = float(
                (actual.float() - expected.float()).norm() / expected.float().norm()
            )
            ref_error = float(
                (actual[: reference.shape[0]].float() - reference).norm()
                / reference.norm()
            )
            reference_errors[name] = ref_error
            if args.dump_output and name.startswith("mega"):
                torch.save(
                    actual.clone().cpu(),
                    args.output / f"output-rows{rows}-rank{rank}.pt",
                )
            print(
                "ACCURACY",
                rank,
                rows,
                name,
                "vs_humming",
                err,
                "vs_fp32",
                ref_error,
                flush=True,
            )
            assert bool(torch.isfinite(actual).all()) and ref_error < 0.05, (
                rank,
                name,
                rows,
                err,
                ref_error,
            )
            errors[name] = err
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with torch.cuda.graph(graph):
                fn()
            graphs[name] = graph
            graph.replay()
            torch.cuda.synchronize()
        if args.ncu:
            name = (
                f"mega_sms{args.sms[0]}"
                if args.ncu_variant == "mega"
                else "humming_ep4"
            )
            deep_gemm.set_num_sms(args.sms[0])
            dist.barrier()
            torch.cuda.cudart().cudaProfilerStart()
            functions[name]()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            break
        samples = {name: [] for name in functions}
        for trial in range(args.rounds):
            for name in list(graphs)[:: 1 if trial % 2 == 0 else -1]:
                dist.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(args.iters):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(
                    start.elapsed_time(end) * 1000 / args.iters, device="cuda"
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                samples[name].append(float(elapsed))
        row = dict(
            rows_per_rank=rows,
            capacity=max(rows, args.capacity),
            experts=args.experts,
            topk=args.topk,
            timing={name: summary(v) for name, v in samples.items()},
            rank=rank,
            relative_l2=errors,
            relative_l2_fp32=reference_errors,
            scope="routed chain including NCCL collectives; excludes router/shared expert",
        )
        records.append(row)
        (args.output / f"rank{rank}.json").write_text(json.dumps(records, indent=2))
        if rank == 0:
            print(
                "CHAIN",
                rows,
                {k: round(v["median_us"], 2) for k, v in row["timing"].items()},
                "L2",
                errors,
                flush=True,
            )
        if args.profile:
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                for name, graph in graphs.items():
                    with torch.profiler.record_function(f"{name}_rows{rows}"):
                        for _ in range(3):
                            graph.replay()
                torch.cuda.synchronize()
            args.profile.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(
                str(
                    args.profile
                    / f"moe-chain-e{args.experts}-rows{rows}-rank{rank}.trace.json.gz"
                )
            )
        del graphs, buffers, functions, expected
        gc.collect()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
