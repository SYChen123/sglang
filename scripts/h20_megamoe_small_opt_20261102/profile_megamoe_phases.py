"""Profile shared-LUT MegaMoE with rank-local GPU clocks and export Chrome traces."""

import argparse
import ast
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from run_dispatch_sfb_ablation import HEADER, make_bench, replace_once


def instrument(source):
    role = "        // WGMMA-output register layout helpers"
    source = replace_once(
        source,
        role,
        """        auto phase_clock = []() {
            uint32_t tick;
            asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(tick) :: "memory");
            return tick;
        };
        const bool phase_leader = lane_idx == 0 and warp_idx_in_wg == 0;
        const uint32_t role_start = phase_clock();
        uint32_t stage_wait_ns = 0, wgmma_wait_ns = 0, l2_ns = 0, scatter_ns = 0;
        uint32_t l2_start = 0, l2_end = 0, scatter_start = 0, scatter_end = 0;
        uint32_t l2_tiles = 0;

"""
        + role,
    )

    anchor = "            auto run_k_stages = [&]<bool kIsL1, uint32_t kNSwap>() {"
    source = replace_once(
        source,
        anchor,
        anchor
        + """
            uint32_t l2_begin = 0;
            if constexpr (not kIsL1) {
                if (phase_leader) {
                    l2_begin = phase_clock();
                    if (l2_tiles++ == 0) l2_start = l2_begin;
                }
            }""",
    )
    early_weights = "constexpr bool kEarlyRegisterWeights" in source
    stage_wait = "                full_barriers[stage_idx]->wait(phase);"
    if early_weights:
        stage_wait = "                if constexpr (not kEarlyRegisterWeights)\n                    full_barriers[stage_idx]->wait(phase);"
    anchor = stage_wait + "\n\n                // Read SF"
    wait_body = """                uint32_t stage_begin = 0;
                if constexpr (not kIsL1) {
                    if (phase_leader) stage_begin = phase_clock();
                }
                full_barriers[stage_idx]->wait(phase);
                if constexpr (not kIsL1) {
                    if (phase_leader) stage_wait_ns += phase_clock() - stage_begin;
                }

                // Read SF"""
    if early_weights:
        wait_body = wait_body.replace(
            "                full_barriers[stage_idx]->wait(phase);",
            "                if constexpr (not kEarlyRegisterWeights)\n                    full_barriers[stage_idx]->wait(phase);",
        ).replace(
            "if constexpr (not kIsL1)",
            "if constexpr (not kIsL1 and not kEarlyRegisterWeights)",
        )
        for barrier in (
            "register_weight_ready[stage_idx].wait(phase);",
            "full_barriers[stage_idx]->wait(phase);",
        ):
            old = (
                "                    if constexpr (kEarlyRegisterWeights)\n                        "
                + barrier
            )
            source = replace_once(
                source,
                old,
                """                    if constexpr (kEarlyRegisterWeights) {
                        uint32_t wait_begin = 0;
                        if constexpr (not kIsL1) {
                            if (phase_leader) wait_begin = phase_clock();
                        }
                        """
                + barrier
                + """
                        if constexpr (not kIsL1) {
                            if (phase_leader) stage_wait_ns += phase_clock() - wait_begin;
                        }
                    }""",
            )
    source = replace_once(source, anchor, wait_body)
    for half in (0, 1):
        anchor = f"                            ptx::warpgroup_wait<0>();\n                            promote_swap_accum({half});"
        source = replace_once(
            source,
            anchor,
            f"""                            const uint32_t wait_begin_{half} = phase_leader ? phase_clock() : 0;
                            ptx::warpgroup_wait<0>();
                            if (phase_leader) wgmma_wait_ns += phase_clock() - wait_begin_{half};
                            promote_swap_accum({half});""",
        )
    # End the L2 compute interval at the common L2 epilogue entry.
    source = replace_once(
        source,
        "        uint32_t l2_tiles = 0;",
        "        uint32_t l2_tiles = 0, l2_begin = 0;",
    )
    source = replace_once(source, "            uint32_t l2_begin = 0;\n", "")
    anchor = "                // ---------------- L2 EPILOGUE: BF16 cast + NVLink scatter ----------------"
    source = replace_once(
        source,
        anchor,
        anchor
        + """
                const uint32_t scatter_begin = phase_leader ? phase_clock() : 0;
                if (phase_leader) {
                    l2_end = scatter_begin;
                    l2_ns += l2_end - l2_begin;
                    if (l2_tiles == 1) scatter_start = scatter_begin;
                }""",
    )
    anchor = """                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
            }
        }, cached_recv_counts);

        // ---------------- COMBINE ----------------"""
    source = replace_once(
        source,
        anchor,
        """                ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx);
                if (phase_leader) {
                    scatter_end = phase_clock();
                    scatter_ns += scatter_end - scatter_begin;
                }
            }
        }, cached_recv_counts);

        const uint32_t scatter_fence_begin = phase_clock();
        // ---------------- COMBINE ----------------""",
    )
    anchor = "        // Sync with dispatch (paired with dispatch's pre-cleanup sync)"
    source = replace_once(
        source,
        anchor,
        "        const uint32_t scatter_fence_end = phase_clock();\n" + anchor,
    )
    anchor = (
        "        constexpr uint32_t kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);"
    )
    source = replace_once(
        source,
        anchor,
        "        const uint32_t combine_begin = phase_clock();\n" + anchor,
    )
    anchor = """    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)"""
    return replace_once(
        source,
        anchor,
        """        ptx::tma_store_wait<0>();
        const uint32_t combine_end = phase_clock();
        // Keep diagnostic printf traffic out of all measured math intervals.
        comm::grid_sync<kNumSMs, kEpilogueGridSyncIndex>(
            workspace, sm_idx, epilogue_thread_idx,
            [&]() { ptx::sync_aligned(kNumEpilogueThreads, kEpilogueFullBarrierIdx); });
        if (phase_leader) {
            printf("MOE_PHASE %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u %u\\n",
                   sym_buffer.rank_idx, num_tokens, sm_idx, epilogue_warp_idx, role_start,
                   l2_start, l2_end, scatter_start, scatter_end, scatter_fence_begin,
                   scatter_fence_end, combine_begin, l2_tiles, l2_ns, stage_wait_ns, wgmma_wait_ns);
            printf("MOE_SCATTER %u %u %u %u %u\\n",
                   sym_buffer.rank_idx, num_tokens, sm_idx, epilogue_warp_idx, scatter_ns);
        }
        if (lane_idx == 0 and sm_idx * kNumEpilogueWarps + epilogue_warp_idx < num_tokens)
            printf("MOE_COMBINE %u %u %u %u %u %u\\n",
                   sym_buffer.rank_idx, num_tokens, sm_idx, epilogue_warp_idx, combine_begin, combine_end);
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)""",
    )


def collect(log, output):
    records = {}
    for match in re.finditer(r"MOE_(PHASE|SCATTER|COMBINE) ((?:\d+ ){3,}\d+)", log):
        values = list(map(int, match[2].split()))
        expected = {"PHASE": 16, "SCATTER": 5, "COMBINE": 6}[match[1]]
        if len(values) != expected:
            continue
        rank, rows, cta, warp, *data = values
        records.setdefault((rows, rank), {}).setdefault(match[1], {})[cta, warp] = data

    summaries = []
    for (rows, rank), group in sorted(records.items()):
        phase = group["PHASE"]
        scatter = group["SCATTER"]
        combine = group["COMBINE"]
        assert len(phase) == len(scatter) == 312 and len(combine) == rows
        origin = next(iter(phase.values()))[0]

        def relative(tick):
            return ((tick - origin + 2**31) % 2**32) - 2**31

        def us(ticks):
            return round(ticks / 1000, 3)

        events = []
        zero = min(relative(p[0]) for p in phase.values())

        def event(name, key, begin, end, **args):
            if begin != 0 and end != 0:
                events.append(
                    dict(
                        name=name,
                        ph="X",
                        pid=rank,
                        tid=key[0] * 8 + key[1],
                        ts=us(relative(begin) - zero),
                        dur=us(relative(end) - relative(begin)),
                        args=args,
                    )
                )

        for key, p in phase.items():
            (
                start,
                l2_start,
                l2_end,
                sc_start,
                sc_end,
                fence_start,
                fence_end,
                comb_start,
                tiles,
                l2,
                stage,
                wgmma,
            ) = p
            event(
                "L2 first-to-last (includes gaps)",
                key,
                l2_start,
                l2_end,
                tiles=tiles,
                total_l2_us=us(l2),
                stage_wait_us=us(stage),
                wgmma_wait_us=us(wgmma),
            )
            event(
                "scatter first-to-last (includes gaps)",
                key,
                sc_start,
                sc_end,
                total_scatter_us=us(scatter[key][0]),
            )
            event("pre-combine NVLink barrier", key, fence_start, fence_end)
            event("dispatch cleanup rendezvous", key, fence_end, comb_start)
        for key, (begin, end) in combine.items():
            event("combine active warp", key, begin, end)

        tail = max(phase, key=lambda key: relative(phase[key][5]))
        p = phase[tail]
        done = max(combine, key=lambda key: relative(combine[key][1]))
        summaries.append(
            dict(
                rows=rows,
                rank=rank,
                last_scatter_cta_warp=tail,
                role_to_last_combine_us=us(relative(combine[done][1]) - zero),
                tail_l2_tiles=p[8],
                tail_l2_us=us(p[9]),
                tail_stage_wait_us=us(p[10]),
                tail_wgmma_wait_us=us(p[11]),
                tail_scatter_us=us(scatter[tail][0]),
                tail_nvlink_barrier_us=us(relative(p[6]) - relative(p[5])),
                last_combine_us=us(
                    relative(combine[done][1]) - relative(combine[done][0])
                ),
            )
        )
        with gzip.open(
            output / f"phases-rows{rows}-rank{rank}.trace.json.gz", "wt"
        ) as trace:
            json.dump(dict(traceEvents=events, displayTimeUnit="ns"), trace)
    assert len(summaries) == 8, summaries
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", required=True, type=Path)
    p.add_argument("--bench", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    package = out / "package"
    shutil.copytree(args.package, package)
    header = package / HEADER
    header.write_text(instrument(header.read_text()))
    bench = make_bench(args.bench.read_text())
    # Warm clocks/workspace with the same kernel, then retain one sample.
    # Separate rank files prevent interleaving of device printf fragments.
    bench = replace_once(
        bench,
        '        expected = functions["humming_ep4"]().clone()',
        """        import ctypes
        stdout = os.dup(1)
        try:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), 1)
                for _ in range(3):
                    functions["mega_sms78"]()
                torch.cuda.synchronize()
                ctypes.CDLL(None).fflush(None)
        finally:
            os.dup2(stdout, 1)
            os.close(stdout)
        dist.barrier()
        functions["mega_sms78"]()
        torch.cuda.synchronize()
        del buffers, functions
        continue
        expected = functions["humming_ep4"]().clone()""",
    )
    bench = replace_once(
        bench,
        '    rank = int(os.environ["LOCAL_RANK"])',
        """    rank = int(os.environ["LOCAL_RANK"])
    with (args.output / f"rank{rank}-device.log").open("w") as rank_log:
        os.dup2(rank_log.fileno(), 1)""",
    )
    ast.parse(bench)
    bench_path = out / "phase_bench.py"
    bench_path.write_text(bench)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=4",
        str(bench_path),
        "--output",
        str(out),
        "--rows",
        "8",
        "32",
        "--sms",
        "78",
        "--capacity",
        "9600",
        "--fragmentpack",
        "--pairpack",
        "--signpack",
        "--scale-jitter",
        "2",
    ]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        PYTHONPATH=str(package),
        DG_SM90_FP4_SMALL_M="16",
        DG_SM90_FP4_PAIR_CTAS="2",
        DG_SM90_FP4_REGISTER_STAGES="6",
        LD_LIBRARY_PATH="/usr/local/cuda/compat/lib:"
        + os.environ.get("LD_LIBRARY_PATH", ""),
    )
    log = out / "profile.log"
    (out / "argv.json").write_text(json.dumps(command, indent=2))
    with log.open("w") as f:
        subprocess.run(
            command,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )
    collect(
        "\n".join((out / f"rank{rank}-device.log").read_text() for rank in range(4)),
        out,
    )


if __name__ == "__main__":
    main()
