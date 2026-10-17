"""Score-only comparison; conversion/gather is tested separately from this timing."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.sm90_deepgemm_prefill import (
    deepgemm_fp8_index_logits_prefill,
)
from sglang.kernels.ops.attention.dsv4.sm90_fp8_prefill import fp8_index_logits_prefill
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def torch_scores(q, weights, keys, lens):
    scores = torch.einsum("mhd,nd->mhn", q, keys)
    scores = (scores.relu() * weights[:, :, None]).sum(1).float()
    return scores.masked_fill(
        torch.arange(keys.shape[0], device=q.device)[None, :] >= lens[:, None],
        -torch.inf,
    )


@marker.parametrize("rows", [32, 128], [32])
@marker.parametrize("width", [4096, 65536], [4096])
@marker.benchmark("impl", ["torch", "triton", "deepgemm"])
def benchmark(rows, width, impl):
    if torch.cuda.get_device_capability()[0] != 9:
        return marker.skip("SM90 prefill scorer")
    q = torch.randn(rows, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    keys = torch.randn(width, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
    lens = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    fn = fp8_index_logits_prefill
    kwargs = {}
    if impl == "torch":
        q, keys = q.bfloat16(), keys.bfloat16()
        fn = torch_scores
    elif impl == "deepgemm":
        fn = deepgemm_fp8_index_logits_prefill
        # Runtime prepares these once per gathered request, outside score tiles.
        kwargs["key_scales"] = torch.ones(width, device="cuda", dtype=torch.float32)
    return marker.do_bench(fn, input_args=(q, weights, keys, lens), input_kwargs=kwargs)


if __name__ == "__main__":
    benchmark.run()
