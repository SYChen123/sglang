"""Run L2 readiness and weight-preparation ablations over the shared-LUT path."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from run_dispatch_sfb_ablation import HEADER, replace_once

BASE_VARIANTS = ("control", "chunk-ready", "early-weights", "combined")
VARIANTS = (
    *BASE_VARIANTS,
    "interleave-decode",
    "early-interleave",
    "mask-backoff",
    "wave-8",
    "wave-12",
    "stages4",
    "producer-lut4",
    "active-wave",
)


def optimize(source, variant):
    if variant == "active-wave":
        return configure_active_wave(source)
    if variant.startswith("wave-"):
        wave = int(variant.split("-")[1])
        return replace_once(
            source,
            "        kNumExpertsPerRank, kNumExpertsPerWave,",
            f"        kNumExpertsPerRank, {wave},",
        )
    if variant == "producer-lut4":
        return move_lut_to_producer(source)
    if variant == "mask-backoff":
        # Keep the original all-chunks-ready protocol but reduce contention
        # from loader warps polling the same readiness mask across CTAs.
        return replace_once(
            source,
            "                while (ptx::ld_acq_gpu(ptr) != expected);",
            "                while (ptx::ld_acq_gpu(ptr) != expected) __nanosleep(64);",
        )
    if variant in ("chunk-ready", "combined"):
        source = replace_once(
            source,
            """            } else {
                constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
                const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                // Each L1 N block sets one bit; total bits = L1_SHAPE_N / BLOCK_N.
                const uint64_t expected = (kNumL1BlockNs >= 64)
                    ? ~0ull : ((1ull << kNumL1BlockNs) - 1ull);
                while (ptx::ld_acq_gpu(ptr) != expected);
            }""",
            """            }""",
        )
        anchor = """                    const uint32_t k_idx = k_block_idx * BLOCK_K;

                    // TMA load A"""
        source = replace_once(
            source,
            anchor,
            """                    const uint32_t k_idx = k_block_idx * BLOCK_K;
                    if (block_phase == sched::BlockPhase::Linear2) {
                        constexpr uint32_t kL1Cols = BLOCK_N / 2;
                        static_assert(BLOCK_K % kL1Cols == 0);
                        static_assert(kNumL1BlockNs <= 64);
                        constexpr uint32_t kTilesPerChunk = BLOCK_K / kL1Cols;
                        static_assert(kTilesPerChunk < 64);
                        const uint64_t expected = ((1ull << kTilesPerChunk) - 1)
                            << (k_block_idx * kTilesPerChunk);
                        const auto ptr = workspace.get_l2_arrival_mask_ptr(pool_block_idx);
                        while ((ptx::ld_acq_gpu(ptr) & expected) != expected);
                    }

                    // TMA load A""",
        )
    if variant in ("early-weights", "combined", "early-interleave"):
        anchor = "    constexpr bool kUseEarlyBDecode = kEarlyBDecode;"
        source = replace_once(
            source,
            anchor,
            anchor
            + """
    constexpr bool kEarlyRegisterWeights = kRegisterWeights and not kUseEarlyBDecode;
    __shared__ uint64_t register_weight_storage[kEarlyRegisterWeights ? kNumStages : 1];
    auto* register_weight_ready = reinterpret_cast<Barrier*>(register_weight_storage);""",
        )
        source = replace_once(
            source,
            "                full_barriers[i]->init(kRegisterWeights ? 3 : (kUseEarlyBDecode ? 1 : 2));",
            """                full_barriers[i]->init(
                    kEarlyRegisterWeights ? 1 : (kRegisterWeights ? 3 : (kUseEarlyBDecode ? 1 : 2)));
                if constexpr (kEarlyRegisterWeights)
                    register_weight_ready[i].init(2);""",
        )
        source = replace_once(
            source,
            "? decode_full_barriers[stage_idx]\n                        : full_barriers[stage_idx];",
            """? decode_full_barriers[stage_idx]
                        : (kEarlyRegisterWeights ? &register_weight_ready[stage_idx]
                                                : full_barriers[stage_idx]);""",
        )
        source = replace_once(
            source,
            """                    } else {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_PACKED_SIZE_PER_STAGE);
                    }""",
            """                    } else if constexpr (kEarlyRegisterWeights) {
                        register_weight_ready[stage_idx].arrive_and_expect_tx(SMEM_B_PACKED_SIZE_PER_STAGE);
                    } else {
                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_PACKED_SIZE_PER_STAGE);
                    }""",
        )
        source = replace_once(
            source,
            "                        if (cute::elect_one_sync()) full_barriers[stage_idx]->arrive();",
            """                        if (cute::elect_one_sync()) {
                            if constexpr (kEarlyRegisterWeights)
                                register_weight_ready[stage_idx].arrive();
                            else
                                full_barriers[stage_idx]->arrive();
                        }""",
        )
        source = replace_once(
            source,
            """                full_barriers[stage_idx]->wait(phase);

                // Read SF""",
            """                if constexpr (not kEarlyRegisterWeights)
                    full_barriers[stage_idx]->wait(phase);

                // Read SF""",
        )
        source = replace_once(
            source,
            """                if constexpr (kRegisterWeights) {
                    #pragma unroll
                    for (uint32_t ki = 0; ki < BLOCK_K / 32; ++ki)
                        load_register_weights(ki * 32, register_weights[ki]);
                }""",
            """                if constexpr (kRegisterWeights) {
                    if constexpr (kEarlyRegisterWeights)
                        register_weight_ready[stage_idx].wait(phase);
                    #pragma unroll
                    for (uint32_t ki = 0; ki < BLOCK_K / 32; ++ki)
                        load_register_weights(ki * 32, register_weights[ki]);
                    if constexpr (kEarlyRegisterWeights)
                        full_barriers[stage_idx]->wait(phase);
                }""",
        )
    if variant in ("interleave-decode", "early-interleave"):
        source = replace_once(
            source,
            """                    #pragma unroll
                    for (uint32_t ki = 0; ki < BLOCK_K / 32; ++ki)
                        load_register_weights(ki * 32, register_weights[ki]);""",
            """                    // In L2, prepare the high K half while the first
                    // group is in flight. Its RS fragments have disjoint storage.
                    constexpr uint32_t kPrepareK = (kIsL1 or not kSwapABL2Active)
                        ? BLOCK_K : BLOCK_K / 2;
                    #pragma unroll
                    for (uint32_t ki = 0; ki < kPrepareK / 32; ++ki)
                        load_register_weights(ki * 32, register_weights[ki]);""",
        )
        anchor = """                            ptx::warpgroup_wait<0>();
                            promote_swap_accum(0);"""
        # Insert after commit, before the low-half accumulator is first read.
        anchor = (
            """                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < kSwapAccum; ++ i)
                                ptx::warpgroup_fence_operand(swap_accum[i]);
"""
            + anchor
        )
        source = replace_once(
            source,
            anchor,
            anchor.replace(
                "                            #pragma unroll",
                """                            if constexpr (kRegisterWeights) {
                                #pragma unroll
                                for (uint32_t ki = BLOCK_K / 64; ki < BLOCK_K / 32; ++ki)
                                    load_register_weights(ki * 32, register_weights[ki]);
                            }
                            #pragma unroll""",
                1,
            ),
        )
    return source


def move_lut_to_producer(source):
    anchor = "    __shared__ uint64_t weight_lut[256];"
    source = replace_once(
        source,
        anchor,
        anchor
        + """
    // Keep the same lookup values. Two producer warps prepare the L2
    // lookup words so the math warp no longer follows dependent SFB loads.
    __shared__ uint64_t l2_lut[kNumStages][4][LOAD_BLOCK_N];""",
    )
    source = replace_once(
        source,
        "                full_barriers[i]->init(kRegisterWeights ? 3 : (kUseEarlyBDecode ? 1 : 2));",
        "                full_barriers[i]->init(kRegisterWeights ? 4 : (kUseEarlyBDecode ? 1 : 2));",
    )
    source = replace_once(
        source,
        "            if (warp_idx == kNumDispatchWarps + kNumFP4TMALoaderWarps) {",
        """            const uint32_t producer = warp_idx - kNumDispatchWarps - kNumFP4TMALoaderWarps;
            if (producer < 2) {""",
    )
    source = replace_once(
        source,
        """                        for (uint32_t row = lane_idx; row < LOAD_BLOCK_N; row += 32) {
                            const uint32_t n_global = n_block_idx * BLOCK_N + row;
                            smem_sfb[stage_idx][row] = __ldg(sfb_base +
                                local_expert_idx * sfb_per_expert +
                                n_global * sfb_k_words + kb);
                        }""",
        """                        for (uint32_t row = producer * 32 + lane_idx; row < LOAD_BLOCK_N; row += 64) {
                            const uint32_t n_global = n_block_idx * BLOCK_N + row;
                            const uint32_t packed_sf = __ldg(sfb_base +
                                local_expert_idx * sfb_per_expert +
                                n_global * sfb_k_words + kb);
                            if (is_l1) {
                                smem_sfb[stage_idx][row] = packed_sf;
                            } else {
                                #pragma unroll
                                for (uint32_t group = 0; group < 4; ++group)
                                    l2_lut[stage_idx][group][row] = weight_lut[(packed_sf >> (group * 8)) & 255];
                            }
                        }""",
    )
    source = replace_once(
        source,
        "auto load_register_weights = [&](uint32_t k_off, uint32_t* a) {",
        "auto load_register_weights = [&]<bool kIsL1>(uint32_t k_off, uint32_t* a) {",
    )
    start = """                const uint32_t sf0 = (smem_sfb[stage_idx][row0] >> (k_off / 32 * 8)) & 255;
                const uint32_t sf1 = (smem_sfb[stage_idx][row1] >> (k_off / 32 * 8)) & 255;
                const uint64_t lut0 = weight_lut[sf0];
                const uint64_t lut1 = weight_lut[sf1];"""
    source = replace_once(
        source,
        start,
        """                const uint32_t group = k_off / 32;
                uint64_t lut0, lut1;
                if constexpr (kIsL1) {
                    lut0 = weight_lut[(smem_sfb[stage_idx][row0] >> (group * 8)) & 255];
                    lut1 = weight_lut[(smem_sfb[stage_idx][row1] >> (group * 8)) & 255];
                } else {
                    lut0 = l2_lut[stage_idx][group][row0];
                    lut1 = l2_lut[stage_idx][group][row1];
                }""",
    )
    return replace_once(
        source,
        "load_register_weights(ki * 32, register_weights[ki]);",
        "load_register_weights.template operator()<kIsL1>(ki * 32, register_weights[ki]);",
    )


def configure_active_wave(source):
    anchor = "    auto scheduler = sched::LegacyMegaMoEScheduler<"
    source = replace_once(
        source,
        anchor,
        """    __shared__ uint32_t wave_bounds[kNumExpertsPerRank];
"""
        + anchor,
    )
    source = replace_once(
        source,
        "        layout::SM90Workspace>(workspace);",
        "        layout::SM90Workspace, true>(workspace, wave_bounds);",
    )
    anchor = """        ptx::sync_unaligned(kNumThreads, kSchedulerCountCacheBarrierIdx);
    };"""
    return replace_once(
        source,
        anchor,
        """        ptx::sync_unaligned(kNumThreads, kSchedulerCountCacheBarrierIdx);
        if (thread_idx == 0) {
            // Fit one L1 wave of M-blocks into the available CTA slots.
            constexpr uint32_t limit = cute::max(1u, kNumSMs / kNumL1BlockNs);
            uint32_t begin = 0, blocks = 0;
            for (uint32_t e = 0; e < kNumExpertsPerRank; ++e) {
                const auto count = math::ceil_div(cached_recv_counts[e], BLOCK_M);
                if (count and blocks and blocks + count > limit) {
                    for (uint32_t i = begin; i < e; ++i) wave_bounds[i] = begin << 16 | e;
                    begin = e;
                    blocks = 0;
                }
                blocks += count;
            }
            for (uint32_t i = begin; i < kNumExpertsPerRank; ++i)
                wave_bounds[i] = begin << 16 | kNumExpertsPerRank;
        }
        ptx::sync_unaligned(kNumThreads, kSchedulerCountCacheBarrierIdx);
    };""",
    )


def patch_scheduler(path):
    source = path.read_text()
    prefix, marker, body = source.partition("struct LegacyMegaMoEScheduler {")
    if not marker or not prefix.endswith("typename WorkspaceT = layout::Workspace>\n"):
        raise ValueError("Legacy scheduler template changed")
    prefix = prefix.removesuffix("typename WorkspaceT = layout::Workspace>\n") + (
        "typename WorkspaceT = layout::Workspace, bool kActiveWaves = false>\n"
    )
    body = replace_once(
        body,
        "    const WorkspaceT& workspace;",
        """    const WorkspaceT& workspace;
    const uint32_t* wave_bounds;""",
    )
    body = replace_once(
        body,
        "    CUTLASS_DEVICE explicit LegacyMegaMoEScheduler(const WorkspaceT& workspace): workspace(workspace) {",
        """    CUTLASS_DEVICE explicit LegacyMegaMoEScheduler(
        const WorkspaceT& workspace, const uint32_t* wave_bounds = nullptr)
        : workspace(workspace), wave_bounds(wave_bounds) {""",
    )
    anchor = "    CUTLASS_DEVICE uint32_t get_wave_expert_end_idx() const {"
    body = replace_once(
        body,
        anchor,
        anchor
        + """
        if constexpr (kActiveWaves) return wave_bounds[current_local_expert_idx] & 0xffff;""",
    )
    body = replace_once(
        body,
        "                set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));",
        """                if constexpr (kActiveWaves)
                    set_expert_idx(wave_bounds[current_local_expert_idx - 1] >> 16);
                else
                    set_expert_idx(math::align<uint32_t, false>(current_local_expert_idx - 1, kNumExpertsPerWave));""",
    )
    path.write_text(prefix + marker + body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=BASE_VARIANTS
    )
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    base_header = (args.package / HEADER).read_text()
    if not all(
        marker in base_header
        for marker in ("SM90ResearchRS", "paired_col", "weight_lut[256]")
    ):
        parser.error("Expected a shared-LUT register-pipeline package")
    failures = []
    for variant in args.variants:
        root = output / variant
        package = root / "package"
        shutil.copytree(args.package, package)
        header = package / HEADER
        header.write_text(optimize(base_header, variant))
        if variant == "active-wave":
            patch_scheduler(
                package / "deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh"
            )
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="0,1,2,3",
            PYTHONPATH=str(package),
            DG_SM90_FP4_SMALL_M="16",
            DG_SM90_FP4_PAIR_CTAS="2",
            DG_SM90_FP4_REGISTER_STAGES="4"
            if variant in ("stages4", "producer-lut4")
            else "6",
            LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:"
            + os.environ.get("LD_LIBRARY_PATH", ""),
        )
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(args.bench.resolve()),
            "--output",
            str(root / "results"),
            "--fragmentpack",
            "--pairpack",
            "--signpack",
            "--scale-jitter",
            "2",
            "--rows",
            "8",
            "32",
            "--capacity",
            "9600",
            "--sms",
            "78",
            "--tp-baseline",
            "--rounds",
            "7",
            "--iters",
            "30",
            "--dump-output",
        ]
        if args.profile:
            cmd += ["--profile", str(root / "traces")]
        (root / "argv.json").write_text(json.dumps(cmd, indent=2))
        try:
            with (root / "bench.log").open("w") as log:
                result = subprocess.run(
                    cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=200
                )
            print(variant, result.returncode, flush=True)
            if result.returncode:
                failures.append(variant)
        except subprocess.TimeoutExpired:
            print(variant, "timeout", flush=True)
            failures.append(variant)
    if failures:
        raise SystemExit("Failed variants: " + ", ".join(failures))


if __name__ == "__main__":
    main()
