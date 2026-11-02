"""Build and benchmark isolated overlays of the SM90 register pipeline."""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ABLATIONS = ("control", "sfb-transposed", "sparse-dispatch", "combined")
VARIANTS = (*ABLATIONS, "shared-lut", "combined-lut")
HEADER = Path("deep_gemm/include/deep_gemm/impls/sm90_fp8_fp4_mega_moe.cuh")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one source anchor: {old}")
    return source.replace(old, new, 1)


def make_kernel(source, variant):
    if variant in ("sfb-transposed", "combined", "combined-lut"):
        # Both SFB readers use the same physical format. The Python API keeps
        # its original contiguous shape, including the packed uint32 dtype.
        old = "n_global * sfb_k_words"
        if source.count(old) != 2:
            raise ValueError("Expected both SFB load sites")
        source = source.replace(
            "const uint32_t sfb_k_words = is_l1 ? kL1SFBKWords : kL2SFBKWords;",
            "const uint32_t sfb_n = is_l1 ? L1_SHAPE_N : L2_SHAPE_N;",
        )
        source = replace_once(
            source,
            old + "\n                            + k_block_idx",
            "k_block_idx * sfb_n + n_global",
        )
        source = replace_once(source, old + " + kb", "kb * sfb_n + n_global")
    if variant in ("sparse-dispatch", "combined", "combined-lut"):
        source = replace_once(source, "!= kNumSMs * kNumRanks", "!= kNumRanks")
        source = replace_once(
            source,
            """        // Stake out per-expert SM offsets via global atomic
        #pragma unroll
        for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
            const uint64_t send_value = (1ull << 32) | static_cast<uint64_t>(smem_expert_count[i]);
            smem_expert_count[i] = static_cast<uint32_t>(
                ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
        }""",
            """        // Only CTAs with routes need to reserve slots. Publish completion
        // once per source rank after the existing grid barrier.
        if (sm_idx * kNumDispatchWarps * kNumTokensPerWarp < num_tokens) {
            #pragma unroll
            for (uint32_t i = thread_idx; i < kNumExperts; i += kNumDispatchThreads) {
                const uint32_t count = smem_expert_count[i];
                if (count != 0)
                    smem_expert_count[i] = static_cast<uint32_t>(ptx::atomic_add(
                        workspace.get_expert_send_count_ptr(i), static_cast<uint64_t>(count)));
            }
        }""",
        )
        source = replace_once(
            source,
            "                    expert_status);",
            "                    (1ull << 32) | expert_status);",
        )
    if variant in ("shared-lut", "combined-lut"):
        source = replace_once(
            source,
            "    if (warp_idx == 0) {\n        // Clean expert-count shared memory",
            """    // Each warp may use several exponents in one MMA fragment.
    // Stage the exact constant table in shared memory so divergent
    // exponent lookups don't serialize through the constant broadcast.
    __shared__ uint64_t weight_lut[256];
    if constexpr (kRegisterWeights) {
        for (uint32_t i = thread_idx; i < 256; i += kNumThreads)
            weight_lut[i] = fp4_decode_detail::pack_scaled_e4m3_lut_from_e8m0_const(i);
    }
    if (warp_idx == 0) {
        // Clean expert-count shared memory""",
        )
        for scale in ("sf0", "sf1"):
            source = replace_once(
                source,
                f"fp4_decode_detail::pack_scaled_e4m3_lut_from_e8m0_const({scale})",
                f"weight_lut[{scale}]",
            )
    return source


def make_bench(source):
    source = replace_once(
        source,
        '    parser.add_argument("--signpack", action="store_true")',
        """    parser.add_argument("--signpack", action="store_true")
    parser.add_argument("--transpose-sfb", action="store_true")
    parser.add_argument("--route-mode", choices=("uniform", "hot-rank"), default="uniform")""",
    )
    source = replace_once(
        source,
        "    del raw\n    gc.collect()",
        """    if args.transpose_sfb:
        prepared = tuple(
            (w, sf.transpose(-2, -1).contiguous().view_as(sf))
            for w, sf in prepared
        )
    del raw
    gc.collect()""",
    )
    return replace_once(
        source,
        '        weights = torch.rand(rows, args.topk, device="cuda")',
        """        if args.route_mode == "hot-rank":
            # Keep unique experts within each token when folding to rank 0.
            ids = torch.rand(rows, local_experts, device="cuda").topk(args.topk).indices.int()
        weights = torch.rand(rows, args.topk, device="cuda")""",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--previous", type=Path, help="Prior experiment directory")
    inputs.add_argument(
        "--package", type=Path, help="Unpacked experimental DeepGEMM wheel"
    )
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=ABLATIONS)
    parser.add_argument("--rows", nargs="+", default=["8", "32", "64", "96", "128"])
    parser.add_argument(
        "--route-mode", choices=("uniform", "hot-rank"), default="uniform"
    )
    parser.add_argument("--scale-jitter", type=int, default=2)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.previous:
        previous = args.previous.resolve()
        package = previous / "deps/sfb-producer"
        default_bench = previous / "bench_small.py"
    else:
        package = args.package.resolve()
        default_bench = Path(__file__).with_name("bench_small.py")
    header = (package / HEADER).read_text()
    if (
        "SM90ResearchRS" not in header
        or "paired_col" not in header
        or "weight_lut[256]" in header
    ):
        parser.error(
            "Expected the register-pipeline package before applying the shared-LUT overlay"
        )
    bench = make_bench((args.bench or default_bench).read_text())
    ast.parse(bench)
    output.mkdir(parents=True, exist_ok=True)
    bench_path = output / "bench_dispatch_sfb.py"
    bench_path.write_text(bench)
    for variant in args.variants:
        variant_dir = output / variant
        variant_package = variant_dir / "package"
        shutil.copytree(package, variant_package)
        (variant_package / HEADER).write_text(make_kernel(header, variant))
        if args.prepare_only:
            continue
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(bench_path),
            "--output",
            str(variant_dir / "results"),
            "--fragmentpack",
            "--pairpack",
            "--signpack",
            "--scale-jitter",
            str(args.scale_jitter),
            "--rows",
            *args.rows,
            "--capacity",
            "9600",
            "--sms",
            "78",
            "--tp-baseline",
            "--rounds",
            "7",
            "--iters",
            "30",
            "--route-mode",
            args.route_mode,
        ]
        if variant in ("sfb-transposed", "combined", "combined-lut"):
            command.append("--transpose-sfb")
        env = dict(
            os.environ,
            PYTHONPATH=str(variant_package),
            CUDA_VISIBLE_DEVICES=args.gpus,
            DG_SM90_FP4_SMALL_M="16",
            DG_SM90_FP4_PAIR_CTAS="2",
            DG_SM90_FP4_REGISTER_STAGES="6",
            LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:"
            + os.environ.get("LD_LIBRARY_PATH", ""),
        )
        (variant_dir / "argv.json").write_text(json.dumps(command, indent=2))
        print(f"Running {variant}", flush=True)
        with (variant_dir / "bench.log").open("w") as log:
            result = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=420
            )
        print(f"{variant} exit={result.returncode}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
