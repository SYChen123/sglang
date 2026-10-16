"""Exact parity against the serving path, including packed cache bytes."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
from sglang.kernels.ops.attention.dsv4.c2_decode_pool import c2_decode_pool
from sglang.kernels.ops.attention.dsv4.c2_epilogue_sm90 import c2_epilogue_sm90
from sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant import (
    rope_tail_fake_quant_fp4,
)
from sglang.kernels.ops.layernorm.rmsnorm_fp32 import rmsnorm_fp32
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def reference(pooled, weight, eps, freqs, slots, group_pos, cache, page_size):
    latent = rmsnorm_fp32(pooled.to(torch.bfloat16), weight, eps)
    gathered = freqs[group_pos]
    quantized = rope_tail_fake_quant_fp4(latent, gathered, 64, compressed_kv=True)
    live = slots > 0
    # The old path races on scratch slot 0. Compare defined live stores and
    # require the new path to leave that scratch slot (and all padding) intact.
    if live.any().item():
        fused_store_cache(
            quantized[live].contiguous(),
            cache,
            slots[live].contiguous(),
            page_size=page_size,
            type="flashmla",
        )
    return latent, gathered


def make_freqs(length=8192):
    return torch.polar(
        torch.ones(length, 32, device="cuda"),
        torch.randn(length, 32, device="cuda"),
    )


def test_exact_latent_freqs_and_entire_cache(rows, page_size, weight_dtype):
    torch.manual_seed(39100)
    pooled = torch.randn(rows, 512, device="cuda")
    original = pooled.clone()
    weight = torch.randn(512, device="cuda", dtype=weight_dtype)
    freqs = make_freqs()
    position = torch.randint(0, 8192, (rows,), device="cuda")
    # Non-contiguous physical slots, page boundaries, padding and incomplete pairs.
    capacity = max(rows * 2, page_size * 2)
    slots = (torch.randperm(capacity - 1, device="cuda")[:rows] + 1).to(torch.int32)
    slots[1::3] = 0
    slots[2::5] = -1
    page_bytes = (584 * page_size + 575) // 576 * 576
    cache = torch.randint(
        0,
        256,
        ((capacity + page_size - 1) // page_size, page_bytes),
        device="cuda",
        dtype=torch.uint8,
    )
    expected_cache = cache.clone()
    expected = reference(
        pooled, weight, 1e-20, freqs, slots, position, expected_cache, page_size
    )
    actual = c2_epilogue_sm90(
        pooled,
        weight,
        1e-20,
        freqs,
        slots,
        position,
        cache.view(torch.float8_e4m3fn),
        page_size,
    )
    for got, wanted in zip(actual, expected):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(pooled, original, rtol=0, atol=0)


def test_rounding_and_scale_extremes(weight_scale):
    torch.manual_seed(39101)
    rows, page_size = 16, 128
    pooled = torch.randn(rows, 512, device="cuda")
    # BF16 ties and their immediate FP32 neighbors exercise finish()'s cast.
    pooled[0].fill_(0)
    pooled[1].fill_(1e-12)
    tie = torch.full((512,), 1.00390625, device="cuda")
    pooled[2] = tie
    pooled[3] = torch.nextafter(tie, torch.full_like(tie, torch.inf))
    pooled[4] = torch.nextafter(tie, torch.full_like(tie, -torch.inf))
    weight = (torch.randn(512, device="cuda") * weight_scale).to(torch.bfloat16)
    freqs = make_freqs()
    slots = torch.arange(1, rows + 1, device="cuda", dtype=torch.int64)
    position = torch.arange(rows, device="cuda", dtype=torch.int32)
    cache = torch.full(
        (1, (584 * page_size + 575) // 576 * 576), 93, device="cuda", dtype=torch.uint8
    )
    expected_cache = cache.clone()
    expected = reference(
        pooled, weight, 1e-20, freqs, slots, position, expected_cache, page_size
    )
    actual = c2_epilogue_sm90(
        pooled, weight, 1e-20, freqs, slots, position, cache, page_size
    )
    for got, wanted in zip(actual, expected):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)


def test_pair_pool_graph_replay_rejection_padding_and_ring_wrap(lengths):
    torch.manual_seed(39102)
    bs, ring, dim, padding, page_size = len(lengths), 8, 512, 3, 128
    live = sum(lengths)
    rows = live + padding
    initial = torch.randn(bs * ring + 1, dim * 2, device="cuda")
    initial[-1, :dim] = 0
    initial[-1, dim:] = -torch.inf
    old_state, new_state = initial.clone(), initial.clone()
    kv, score = (torch.randn(rows, dim, device="cuda") for _ in range(2))
    req = torch.tensor(
        [r for r, n in enumerate(lengths) for _ in range(n)] + [0] * padding,
        device="cuda",
        dtype=torch.int64,
    )
    base_pos = torch.tensor(
        [p for n in lengths for p in range(n)] + [0] * padding,
        device="cuda",
        dtype=torch.int64,
    )
    pos = base_pos.clone()
    raw_loc = torch.empty(rows, device="cuda", dtype=torch.int64)
    out_loc = torch.empty_like(raw_loc)
    weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    freqs = make_freqs()
    initial_cache = torch.full(
        (bs * 2 + 2, (584 * page_size + 575) // 576 * 576),
        71,
        device="cuda",
        dtype=torch.uint8,
    )
    old_cache, new_cache = initial_cache.clone(), initial_cache.clone()

    def metadata(advance):
        pos.copy_(base_pos + advance)
        raw_loc.copy_((req + 1) * 512 + pos)
        raw_loc[live:] = 0
        out_loc.copy_(torch.where(pos % 2 == 1, raw_loc // 2, -1))

    def run(state, cache, epilogue):
        pooled, group_pos, slots = c2_decode_pool(
            kv,
            score,
            pos,
            raw_loc,
            out_loc,
            req,
            state[:, :dim],
            state[:, dim:],
            state.shape[0] - 1,
            ring_size=ring,
        )
        return epilogue(
            pooled, weight, 1e-20, freqs, slots, group_pos, cache, page_size
        )

    metadata(0)
    run(new_state, new_cache, c2_epilogue_sm90)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(new_state, new_cache, c2_epilogue_sm90)
    new_state.copy_(initial)
    new_cache.copy_(initial_cache)
    for advance in (0, 1, 3, 6, 7, 8, 13, 16):
        kv.normal_()
        score.normal_()
        metadata(advance)
        expected = run(old_state, old_cache, reference)
        graph.replay()
        for got, wanted in zip(actual, expected):
            torch.testing.assert_close(got, wanted, rtol=0, atol=0)
        torch.testing.assert_close(new_cache, old_cache, rtol=0, atol=0)
        torch.testing.assert_close(new_state, old_state, rtol=0, atol=0)


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.cuda is not None
    and torch.cuda.get_device_capability()[0] >= 9,
    "requires SM90+ CUDA",
)
class TestC2EpilogueNumerics(CustomTestCase):
    def test_exact_bytes_for_shapes_and_dtypes(self):
        for rows in (0, 1, 6, 12, 64, 384):
            for page_size in (64, 128):
                for weight_dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(rows=rows, page=page_size, dtype=weight_dtype):
                        test_exact_latent_freqs_and_entire_cache(
                            rows, page_size, weight_dtype
                        )

    def test_rounding_and_scale_extremes(self):
        for scale in (0.0, 2.0**-16, 1.0, 1024.0):
            with self.subTest(scale=scale):
                test_rounding_and_scale_extremes(scale)

    def test_pool_and_graph_ring_wrap(self):
        for lengths in ((1, 1, 1), (6,), (6, 6), (1, 3, 6)):
            with self.subTest(lengths=lengths):
                test_pair_pool_graph_replay_rejection_padding_and_ring_wrap(lengths)


if __name__ == "__main__":
    unittest.main()
