"""Measure L2 producer work and exposed B readiness with rank-local GPU clocks."""

import gzip
import json
import os
import re

import profile_megamoe_phases as phases
from run_dispatch_sfb_ablation import replace_once


def instrument(source):
    anchor = "    constexpr bool kUseEarlyBDecode = kEarlyBDecode;"
    source = replace_once(
        source,
        anchor,
        anchor
        + """
    static_assert(kRegisterWeights and not kUseEarlyBDecode);
    __shared__ uint64_t profile_b_storage[kNumStages];
    __shared__ uint32_t profile_b_issue[kNumStages];
    __shared__ volatile uint32_t profile_a_wait, profile_sfb_copy;
    auto* profile_b = reinterpret_cast<Barrier*>(profile_b_storage);
    auto profile_clock = []() {
        uint32_t tick;
        asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(tick) :: "memory");
        return tick;
    };""",
    )
    source = replace_once(
        source,
        "    if (warp_idx == 0) {\n        // Clean expert-count shared memory",
        """    if (warp_idx == 0) {
        if (lane_idx == 0) {
            profile_a_wait = 0;
            profile_sfb_copy = 0;
        }
        // Clean expert-count shared memory""",
    )
    source = replace_once(
        source,
        "                full_barriers[i]->init(kRegisterWeights ? 3 : (kUseEarlyBDecode ? 1 : 2));",
        """                full_barriers[i]->init(2);
                profile_b[i].init(1);""",
    )
    source = replace_once(
        source,
        "                while (ptx::ld_acq_gpu(ptr) != expected);",
        """                const uint32_t tick = lane_idx == 0 ? profile_clock() : 0;
                while (ptx::ld_acq_gpu(ptr) != expected);
                if (lane_idx == 0) profile_a_wait = profile_a_wait + (profile_clock() - tick);""",
    )
    source = replace_once(
        source,
        """                    auto b_full_barrier = kUseEarlyBDecode
                        ? decode_full_barriers[stage_idx]
                        : full_barriers[stage_idx];""",
        """                    profile_b_issue[stage_idx] = profile_clock();
                    auto b_full_barrier = &profile_b[stage_idx];""",
    )
    source = replace_once(
        source,
        "                        full_barriers[stage_idx]->arrive_and_expect_tx(SMEM_B_PACKED_SIZE_PER_STAGE);",
        "                        profile_b[stage_idx].arrive_and_expect_tx(SMEM_B_PACKED_SIZE_PER_STAGE);",
    )
    anchor = """                        empty_barriers[stage_idx]->wait(phase ^ 1);
                        #pragma unroll
                        for (uint32_t row = lane_idx; row < LOAD_BLOCK_N; row += 32)"""
    source = replace_once(
        source,
        anchor,
        anchor.replace(
            "                        #pragma unroll",
            """
                        const uint32_t tick = not is_l1 and lane_idx == 0 ? profile_clock() : 0;
                        #pragma unroll""",
        ),
    )
    source = replace_once(
        source,
        "                        if (cute::elect_one_sync()) full_barriers[stage_idx]->arrive();",
        """                        if (not is_l1 and lane_idx == 0)
                            profile_sfb_copy = profile_sfb_copy + (profile_clock() - tick);
                        if (cute::elect_one_sync()) full_barriers[stage_idx]->arrive();""",
    )
    anchor = "        // WGMMA-output register layout helpers"
    source = replace_once(
        source,
        anchor,
        """        const bool record = warp_idx_in_wg == 0 and lane_idx == 0;
        const uint32_t role_start = profile_clock();
        uint32_t b_elapsed = 0, b_wait = 0, asf_wait = 0, unpack = 0;
"""
        + anchor,
    )
    anchor = "                full_barriers[stage_idx]->wait(phase);\n\n                // Read SF"
    source = replace_once(
        source,
        anchor,
        """                const uint32_t b_begin = not kIsL1 and record ? profile_clock() : 0;
                profile_b[stage_idx].wait(phase);
                const uint32_t b_done = not kIsL1 and record ? profile_clock() : 0;
                full_barriers[stage_idx]->wait(phase);
                if constexpr (not kIsL1) {
                    if (record) {
                        b_wait += b_done - b_begin;
                        b_elapsed += b_done - profile_b_issue[stage_idx];
                        asf_wait += profile_clock() - b_done;
                    }
                }

                // Read SF""",
    )
    anchor = """                    #pragma unroll
                    for (uint32_t ki = 0; ki < BLOCK_K / 32; ++ki)
                        load_register_weights(ki * 32, register_weights[ki]);"""
    source = replace_once(
        source,
        anchor,
        """                    const uint32_t tick = not kIsL1 and record ? profile_clock() : 0;
"""
        + anchor
        + """
                    if constexpr (not kIsL1) {
                        if (record) unpack += profile_clock() - tick;
                    }""",
    )
    if os.environ.get("MEGAMOE_PROFILE_B_OBSERVER") == "1":
        source = add_b_observer(source)
    source = replace_once(
        source,
        "        // ---------------- COMBINE ----------------",
        "        const uint32_t scatter_done = profile_clock();\n        // ---------------- COMBINE ----------------",
    )
    anchor = """    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)"""
    source = replace_once(
        source,
        anchor,
        """        const uint32_t end = profile_clock();
        comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
            workspace, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); });
        if (record)
            printf("L2_SUPPLY %u %u %u %u %u %u %u %u %u %u %u %u %u\\n",
                   sym_buffer.rank_idx, num_tokens, sm_idx, epilogue_warp_idx,
                   role_start, scatter_done, end, profile_a_wait, b_elapsed, profile_sfb_copy,
                   unpack, b_wait, asf_wait);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)""",
    )

    if os.environ.get("MEGAMOE_PROFILE_B_OBSERVER") == "1":
        source = replace_once(
            source,
            "profile_a_wait, b_elapsed, profile_sfb_copy",
            "profile_a_wait, profile_b_latency, profile_sfb_copy",
        )
    return source


def add_b_observer(source):
    """Observe completed TMA on an otherwise idle warp, before math demand."""
    source = replace_once(
        source,
        "profile_a_wait, profile_sfb_copy;",
        "profile_a_wait, profile_sfb_copy, profile_b_latency;",
    )
    source = replace_once(
        source,
        "            profile_a_wait = 0;",
        "            profile_a_wait = 0;\n            profile_b_latency = 0;",
    )
    source = replace_once(
        source,
        "                empty_barriers[i]->init(kNumEpilogueWarps);",
        "                empty_barriers[i]->init(kNumEpilogueWarps + 1);",
    )
    anchor = """                }, cached_recv_counts);
            }
        } else {
            const uint32_t non_epilogue_warp_idx"""
    return replace_once(
        source,
        anchor,
        """                }, cached_recv_counts);
            } else if (warp_idx == kNumDispatchWarps + kNumFP4TMALoaderWarps + 1) {
                static_assert(kNumMMANonEpilogueWarps == 4 and kNumFP4TMALoaderWarps == 2);
                sm90_fp8_fp4_mega_moe_for_each_cached_block<
                    kNumExpertsPerRank, kNumExpertsPerLane,
                    L1_SHAPE_K / BLOCK_K, L2_SHAPE_K / BLOCK_K>(
                    scheduler, [&](const sched::BlockPhase& phase_of_tile,
                                   const uint32_t&, const uint32_t& num_k_blocks,
                                   const uint32_t&, const uint32_t&) {
                    for (uint32_t kb = 0; kb < num_k_blocks; advance_pipeline(kb)) {
                        profile_b[stage_idx].wait(phase);
                        if (lane_idx == 0) {
                            if (phase_of_tile == sched::BlockPhase::Linear2)
                                profile_b_latency = profile_b_latency +
                                    (profile_clock() - profile_b_issue[stage_idx]);
                            empty_barriers[stage_idx]->arrive();
                        }
                        __syncwarp();
                    }
                }, cached_recv_counts);
            }
        } else {
            const uint32_t non_epilogue_warp_idx""",
    )


def collect(log, output):
    groups = {}
    for match in re.finditer(r"L2_SUPPLY ((?:\d+ ){12}\d+)", log):
        rank, rows, cta, warp, *vals = map(int, match[1].split())
        groups.setdefault((rows, rank), {})[cta, warp] = vals
    assert len(groups) == 8, list(groups)
    summaries = []
    b_key = (
        "b_issue_to_observer_us"
        if os.environ.get("MEGAMOE_PROFILE_B_OBSERVER") == "1"
        else "b_issue_to_observed_us"
    )
    keys = (
        "a_publish_wait_us",
        b_key,
        "sfb_copy_us",
        "fp4_unpack_us",
        "b_first_wait_us",
        "a_sfb_after_b_wait_us",
    )
    for (rows, rank), group in sorted(groups.items()):
        assert len(group) == 312, (rows, rank, len(group))
        origin = next(iter(group.values()))[0]

        def rel(tick):
            return ((tick - origin + 2**31) % 2**32) - 2**31

        zero = min(rel(values[0]) for values in group.values())
        events = []
        for (cta, warp), (begin, scatter, end, *stats) in group.items():
            events.append(
                dict(
                    name="Math through scatter",
                    ph="X",
                    pid=rank,
                    tid=cta * 8 + warp,
                    ts=(rel(begin) - zero) / 1000,
                    dur=(rel(scatter) - rel(begin)) / 1000,
                    args=dict(zip(keys, (v / 1000 for v in stats))),
                )
            )
            events.append(
                dict(
                    name="Post-scatter through combine issue",
                    ph="X",
                    pid=rank,
                    tid=cta * 8 + warp,
                    ts=(rel(scatter) - zero) / 1000,
                    dur=(rel(end) - rel(scatter)) / 1000,
                )
            )
        cta, warp = max(group, key=lambda key: rel(group[key][1]))
        summaries.append(
            dict(
                rows=rows,
                rank=rank,
                tail_cta=cta,
                tail_warp=warp,
                tail_scatter_us=(rel(group[cta, warp][1]) - zero) / 1000,
                **dict(zip(keys, (v / 1000 for v in group[cta, warp][3:]))),
            )
        )
        with gzip.open(
            output / f"supply-rows{rows}-rank{rank}.trace.json.gz", "wt"
        ) as f:
            json.dump(dict(traceEvents=events, displayTimeUnit="ns"), f)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    phases.instrument = instrument
    phases.collect = collect
    phases.main()
