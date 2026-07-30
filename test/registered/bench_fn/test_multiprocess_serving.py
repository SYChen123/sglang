import asyncio
import json
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.benchmark.arrival_schedule import generate_arrival_schedule
from sglang.benchmark.multiprocess_serving import (
    GlobalAdmissionState,
    _build_worker_command,
    _flush_cache,
    _resolve_base_url,
    _run_global_schedule,
    aggregate_results,
    split_integer,
)
from sglang.benchmark.serving import (
    RequestFuncOutput,
    _shard_input_requests,
    calculate_metrics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeCoordinatorSocket:
    def __init__(self):
        self.incoming = deque()
        self.dispatched = []
        self.inflight = 0
        self.max_inflight = 0

    async def send_multipart(self, frames):
        worker_index = int(frames[0].decode("ascii"))
        message = json.loads(frames[1].decode("utf-8"))
        if message["type"] != "dispatch":
            return
        global_request_id = message["global_request_id"]
        self.dispatched.append(global_request_id)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        self.incoming.append(
            [
                frames[0],
                json.dumps(
                    {
                        "type": "done",
                        "run_id": message["run_id"],
                        "worker_index": worker_index,
                        "global_request_id": global_request_id,
                        "success": True,
                    }
                ).encode("utf-8"),
            ]
        )

    async def poll(self, timeout):
        await asyncio.sleep(0)
        return bool(self.incoming)

    async def recv_multipart(self):
        self.inflight -= 1
        return self.incoming.popleft()


def _child_result(
    *,
    global_request_ids,
    benchmark_start_time,
    benchmark_end_time,
    successes,
    latencies,
    start_times,
    ttfts,
    itls,
    metric_itls,
    output_lens,
    total_input,
    total_output,
):
    duration = benchmark_end_time - benchmark_start_time
    completed = sum(successes)
    count = len(global_request_ids)
    return {
        "tag": None,
        "backend": "sglang",
        "dataset_name": "generated-shared-prefix",
        "benchmark_start_time": benchmark_start_time,
        "benchmark_end_time": benchmark_end_time,
        "duration": duration,
        "global_request_ids": global_request_ids,
        "input_lens": [10] * count,
        "output_lens": output_lens,
        "successes": successes,
        "latencies": latencies,
        "start_times": start_times,
        "ttfts": ttfts,
        "itls": itls,
        "_raw_metric_itls": metric_itls,
        "_retokenized_metric_itls": metric_itls,
        "errors": [None if success else "failed" for success in successes],
        "completed": completed,
        "total_input_tokens": total_input,
        "total_input_text_tokens": total_input,
        "total_input_vision_tokens": 0,
        "total_output_tokens": total_output,
        "total_output_tokens_retokenized": total_output,
        "request_throughput": completed / duration,
        "output_throughput": total_output / duration,
        "mean_ttft_ms": 0,
        "mean_tpot_ms": 0,
        "mean_e2e_latency_ms": 0,
        "accept_length": None,
        "server_info": None,
    }


class TestMultiprocessServing(CustomTestCase):
    @staticmethod
    def _make_output(
        *,
        start_time,
        latency,
        ttft,
        itls,
        output_len,
        success=True,
    ):
        return RequestFuncOutput(
            generated_text=" ".join(["token"] * output_len),
            success=success,
            latency=latency,
            ttft=ttft,
            itl=itls,
            output_len=output_len,
            start_time=start_time,
        )

    def test_arrival_schedule_numbers_are_absolute_offsets(self):
        schedule = generate_arrival_schedule(
            num_requests=37,
            request_rate=15.0,
            seed=1234,
        )

        self.assertEqual(schedule[0], 0.0)
        self.assertTrue(
            all(left < right for left, right in zip(schedule, schedule[1:]))
        )
        self.assertEqual(
            schedule,
            generate_arrival_schedule(37, request_rate=15.0, seed=1234),
        )
        self.assertEqual(
            generate_arrival_schedule(4, request_rate=float("inf"), seed=7),
            [0.0, 0.0, 0.0, 0.0],
        )

    def test_unlimited_admission_does_not_create_a_concurrency_gate(self):
        state = GlobalAdmissionState(total_requests=6, max_concurrency=None)
        schedule = [0.0] * 6
        state.enqueue_due(0.0, schedule)

        dispatches = state.take_dispatchable(worker_count=3)

        self.assertEqual(
            dispatches,
            [
                (0, 0, 0),
                (1, 1, 0),
                (2, 2, 0),
                (3, 0, 1),
                (4, 1, 1),
                (5, 2, 1),
            ],
        )
        self.assertEqual(len(state.active), 6)
        self.assertFalse(state.pending)

    def test_global_concurrency_is_fifo_and_borrowed_across_workers(self):
        state = GlobalAdmissionState(total_requests=6, max_concurrency=2)
        state.enqueue_due(0.0, [0.0] * 6)

        first = state.take_dispatchable(worker_count=4)
        self.assertEqual(first, [(0, 0, 0), (1, 1, 0)])
        self.assertEqual(len(state.active), 2)

        state.complete(global_request_id=0, worker_index=0)
        second = state.take_dispatchable(worker_count=4)
        self.assertEqual(second, [(2, 2, 0)])
        self.assertEqual(len(state.active), 2)

        state.complete(global_request_id=1, worker_index=1)
        third = state.take_dispatchable(worker_count=4)
        self.assertEqual(third, [(3, 3, 0)])
        self.assertEqual(len(state.active), 2)

    def test_global_concurrency_can_be_less_than_worker_count(self):
        state = GlobalAdmissionState(total_requests=3, max_concurrency=1)
        state.enqueue_due(0.0, [0.0, 0.0, 0.0])

        self.assertEqual(state.take_dispatchable(8), [(0, 0, 0)])
        state.complete(0, 0)
        self.assertEqual(state.take_dispatchable(8), [(1, 1, 0)])

    def test_schedule_runner_applies_one_global_concurrency_limit(self):
        socket = _FakeCoordinatorSocket()
        workers = [
            SimpleNamespace(index=index, process=MagicMock()) for index in range(4)
        ]
        for worker in workers:
            worker.process.poll.return_value = None

        asyncio.run(
            _run_global_schedule(
                socket,
                workers,
                run_id="test",
                schedule=[0.0] * 12,
                start_time=0.0,
                max_concurrency=3,
            )
        )

        self.assertEqual(socket.dispatched, list(range(12)))
        self.assertEqual(socket.max_inflight, 3)

    def test_schedule_runner_has_no_gate_without_max_concurrency(self):
        socket = _FakeCoordinatorSocket()
        workers = [
            SimpleNamespace(index=index, process=MagicMock()) for index in range(2)
        ]
        for worker in workers:
            worker.process.poll.return_value = None

        asyncio.run(
            _run_global_schedule(
                socket,
                workers,
                run_id="test",
                schedule=[0.0] * 7,
                start_time=0.0,
                max_concurrency=None,
            )
        )

        self.assertEqual(socket.dispatched, list(range(7)))
        self.assertEqual(socket.max_inflight, 7)

    def test_admission_rejects_duplicate_or_wrong_worker_completion(self):
        state = GlobalAdmissionState(total_requests=1, max_concurrency=1)
        state.enqueue_due(0.0, [0.0])
        state.take_dispatchable(worker_count=2)

        with self.assertRaisesRegex(RuntimeError, "expected worker 0"):
            state.complete(0, 1)
        state.complete(0, 0)
        with self.assertRaisesRegex(RuntimeError, "duplicate completion"):
            state.complete(0, 0)

    def test_worker_command_has_no_nested_launcher_or_static_concurrency_split(self):
        command = _build_worker_command(
            [
                "--backend",
                "sglang",
                "--num-processes",
                "8",
                "--num-prompts",
                "2048",
                "--request-rate",
                "15",
                "--max-concurrency",
                "448",
                "--output-file",
                "aggregate.jsonl",
            ],
            total_num_prompts=2048,
            max_concurrency=448,
            request_rate=15,
            warmup_requests=0,
            result_file=Path("worker.jsonl"),
            prepare_only=False,
        )

        self.assertNotIn("--num-processes", command)
        self.assertEqual(command.count("--num-prompts"), 1)
        self.assertEqual(command[command.index("--num-prompts") + 1], "2048")
        self.assertEqual(command[command.index("--request-rate") + 1], "15")
        self.assertEqual(command[command.index("--max-concurrency") + 1], "448")
        self.assertEqual(command[command.index("--output-file") + 1], "worker.jsonl")

    def test_prepare_only_worker_does_not_require_rate_or_result_file(self):
        command = _build_worker_command(
            [
                "--model",
                "/models/test",
                "--dataset-name",
                "generated-shared-prefix",
            ],
            total_num_prompts=16,
            max_concurrency=None,
            request_rate=float("inf"),
            warmup_requests=0,
            result_file=Path("worker.jsonl"),
            prepare_only=True,
        )

        self.assertIn("--prepare-only", command)
        self.assertNotIn("--request-rate", command)
        self.assertNotIn("--warmup-requests", command)
        self.assertNotIn("--output-file", command)

    def test_split_integer_matches_stride_shard_sizes(self):
        self.assertEqual(split_integer(10, 3), [4, 3, 3])
        self.assertEqual(split_integer(448, 8), [56] * 8)

    def test_backend_specific_default_ports_and_cache_flush(self):
        self.assertEqual(
            _resolve_base_url(
                SimpleNamespace(
                    base_url=None,
                    backend="vllm-chat",
                    host="0.0.0.0",
                    port=None,
                )
            ),
            "http://0.0.0.0:8000",
        )

        response = MagicMock()
        response.__enter__.return_value.status = 200
        with patch(
            "sglang.benchmark.multiprocess_serving.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            _flush_cache("http://server:8000", "vllm-chat")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://server:8000/reset_prefix_cache")

    def test_request_shards_are_disjoint_and_complete(self):
        requests = list(range(10))
        shards = []
        for index in range(3):
            with patch.dict(
                "os.environ",
                {
                    "SGLANG_BENCH_NUM_SHARDS": "3",
                    "SGLANG_BENCH_SHARD_INDEX": str(index),
                },
                clear=False,
            ):
                shards.append(_shard_input_requests(requests))

        self.assertEqual(shards, [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]])
        self.assertEqual(sorted(value for shard in shards for value in shard), requests)

    def test_aggregate_recomputes_request_level_metrics_and_global_order(self):
        child_results = [
            _child_result(
                global_request_ids=[0],
                benchmark_start_time=100.0,
                benchmark_end_time=104.0,
                successes=[True],
                latencies=[2.0],
                start_times=[100.0],
                ttfts=[0.5],
                itls=[[0.75, 0.75]],
                metric_itls=[[0.75, 0.75]],
                output_lens=[3],
                total_input=10,
                total_output=3,
            ),
            _child_result(
                global_request_ids=[1, 2],
                benchmark_start_time=100.0,
                benchmark_end_time=104.5,
                successes=[True, False],
                latencies=[1.0, 0.0],
                start_times=[100.1, 0.0],
                ttfts=[0.2, 0.0],
                itls=[[0.8], []],
                metric_itls=[[0.8], []],
                output_lens=[2, 0],
                total_input=20,
                total_output=2,
            ),
        ]

        result = aggregate_results(
            child_results,
            total_num_prompts=3,
            total_concurrency=2,
            total_request_rate=20,
            include_details=True,
        )

        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["failed"], 1)
        self.assertAlmostEqual(result["duration"], 4.5)
        self.assertAlmostEqual(result["request_throughput"], 2 / 4.5)
        self.assertAlmostEqual(result["output_throughput"], 5 / 4.5)
        self.assertAlmostEqual(result["mean_e2e_latency_ms"], 1500)
        self.assertAlmostEqual(result["mean_ttft_ms"], 350)
        self.assertAlmostEqual(result["mean_tpot_ms"], 775)
        self.assertAlmostEqual(result["concurrency"], 3 / 4.5)
        self.assertEqual(result["max_output_tokens_per_s"], 2)
        self.assertEqual(result["max_concurrent_requests"], 2)
        self.assertEqual(result["latencies"], [2.0, 1.0, 0.0])

    def test_details_are_restored_by_global_request_id(self):
        child_results = [
            _child_result(
                global_request_ids=[0, 2],
                benchmark_start_time=100.0,
                benchmark_end_time=102.0,
                successes=[True, True],
                latencies=[1.0, 3.0],
                start_times=[100.0, 100.2],
                ttfts=[0.1, 0.3],
                itls=[[], []],
                metric_itls=[[], []],
                output_lens=[1, 1],
                total_input=20,
                total_output=2,
            ),
            _child_result(
                global_request_ids=[1, 3],
                benchmark_start_time=100.0,
                benchmark_end_time=104.0,
                successes=[True, True],
                latencies=[2.0, 4.0],
                start_times=[100.1, 100.3],
                ttfts=[0.2, 0.4],
                itls=[[], []],
                metric_itls=[[], []],
                output_lens=[1, 1],
                total_input=20,
                total_output=2,
            ),
        ]

        result = aggregate_results(
            child_results,
            total_num_prompts=4,
            total_concurrency=None,
            total_request_rate=10,
            include_details=True,
        )

        self.assertEqual(result["latencies"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["ttfts"], [0.1, 0.2, 0.3, 0.4])

    def test_peak_metrics_exactly_match_single_process_calculation(self):
        outputs = [
            self._make_output(
                start_time=100.10,
                latency=2.30,
                ttft=0.20,
                itls=[0.40, 0.60, 0.70],
                output_len=4,
            ),
            self._make_output(
                start_time=100.85,
                latency=1.75,
                ttft=0.10,
                itls=[0.15, 0.90],
                output_len=3,
            ),
            self._make_output(
                start_time=101.35,
                latency=0.80,
                ttft=0.05,
                itls=[],
                output_len=1,
            ),
            self._make_output(
                start_time=100.50,
                latency=0.25,
                ttft=0.0,
                itls=[],
                output_len=0,
                success=False,
            ),
        ]

        tokenizer = MagicMock()
        tokenizer.encode.side_effect = lambda text, **_: text.split()
        single_metrics, _ = calculate_metrics(
            input_requests=None,
            outputs=outputs,
            dur_s=3.0,
            tokenizer=tokenizer,
            backend="sglang",
        )

        child_results = [
            _child_result(
                global_request_ids=[0, 3],
                benchmark_start_time=100.0,
                benchmark_end_time=102.5,
                successes=[outputs[0].success, outputs[3].success],
                latencies=[outputs[0].latency, outputs[3].latency],
                start_times=[outputs[0].start_time, outputs[3].start_time],
                ttfts=[outputs[0].ttft, outputs[3].ttft],
                itls=[outputs[0].itl, outputs[3].itl],
                metric_itls=[outputs[0].itl, outputs[3].itl],
                output_lens=[outputs[0].output_len, outputs[3].output_len],
                total_input=0,
                total_output=outputs[0].output_len,
            ),
            _child_result(
                global_request_ids=[1, 2],
                benchmark_start_time=100.0,
                benchmark_end_time=103.0,
                successes=[outputs[1].success, outputs[2].success],
                latencies=[outputs[1].latency, outputs[2].latency],
                start_times=[outputs[1].start_time, outputs[2].start_time],
                ttfts=[outputs[1].ttft, outputs[2].ttft],
                itls=[outputs[1].itl, outputs[2].itl],
                metric_itls=[outputs[1].itl, outputs[2].itl],
                output_lens=[outputs[1].output_len, outputs[2].output_len],
                total_input=0,
                total_output=outputs[1].output_len + outputs[2].output_len,
            ),
        ]
        aggregate = aggregate_results(
            child_results,
            total_num_prompts=len(outputs),
            total_concurrency=None,
            total_request_rate=10,
        )

        self.assertEqual(
            aggregate["max_output_tokens_per_s"],
            single_metrics.max_output_tokens_per_s,
        )
        self.assertEqual(
            aggregate["max_concurrent_requests"],
            single_metrics.max_concurrent_requests,
        )

    def test_aggregate_rejects_mismatched_metric_details(self):
        child = _child_result(
            global_request_ids=[0],
            benchmark_start_time=100.0,
            benchmark_end_time=101.0,
            successes=[True],
            latencies=[1.0],
            start_times=[100.0],
            ttfts=[0.2],
            itls=[[0.8]],
            metric_itls=[],
            output_lens=[2],
            total_input=10,
            total_output=2,
        )
        with self.assertRaisesRegex(ValueError, "detail arrays have different lengths"):
            aggregate_results(
                [child],
                total_num_prompts=1,
                total_concurrency=1,
                total_request_rate=1,
            )


if __name__ == "__main__":
    unittest.main()
