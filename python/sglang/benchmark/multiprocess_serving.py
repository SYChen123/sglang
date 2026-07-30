# SPDX-License-Identifier: Apache-2.0

"""Parent-side orchestration for multi-process serving benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np
import zmq
import zmq.asyncio

from sglang.benchmark.arrival_schedule import generate_arrival_schedule
from sglang.benchmark.multiprocess_control import (
    CONTROL_ENDPOINT_ENV,
    CONTROL_RUN_ID_ENV,
    CONTROL_WORKER_INDEX_ENV,
)
from sglang.benchmark.peak_metrics import RequestTiming, calculate_peak_metrics

_PARENT_VALUE_OPTIONS = {
    "--max-concurrency",
    "--multiprocess-barrier-timeout-sec",
    "--multiprocess-child-output-dir",
    "--num-processes",
    "--num-prompts",
    "--output-file",
    "--request-rate",
    "--warmup-requests",
}
_PARENT_FLAG_OPTIONS = {
    "--disable-tqdm",
    "--flush-cache",
    "--output-details",
    "--prepare-only",
}
_DETAIL_KEYS = (
    "input_lens",
    "output_lens",
    "successes",
    "latencies",
    "start_times",
    "ttfts",
    "itls",
    "errors",
)
_OPTIONAL_DETAIL_KEYS = (
    "generated_texts",
    "cached_tokens",
    "cached_tokens_details",
)
_INTERNAL_DETAIL_KEYS = (
    "_raw_metric_itls",
    "_retokenized_metric_itls",
)


@dataclass
class Worker:
    index: int
    num_prompts: int
    process: subprocess.Popen
    log_file: Path
    result_file: Path
    log_handle: Any


@dataclass
class GlobalAdmissionState:
    """The single FIFO admission state shared by all load-generator workers."""

    total_requests: int
    max_concurrency: Optional[int]
    next_arrival: int = 0
    pending: Deque[int] = field(default_factory=deque)
    active: Dict[int, int] = field(default_factory=dict)
    completed: set[int] = field(default_factory=set)

    def enqueue_due(self, elapsed: float, schedule: Sequence[float]) -> None:
        while (
            self.next_arrival < self.total_requests
            and schedule[self.next_arrival] <= elapsed
        ):
            self.pending.append(self.next_arrival)
            self.next_arrival += 1

    def take_dispatchable(self, worker_count: int) -> List[tuple[int, int, int]]:
        dispatches = []
        while self.pending and (
            self.max_concurrency is None or len(self.active) < self.max_concurrency
        ):
            global_request_id = self.pending.popleft()
            worker_index = global_request_id % worker_count
            local_request_id = global_request_id // worker_count
            self.active[global_request_id] = worker_index
            dispatches.append((global_request_id, worker_index, local_request_id))
        return dispatches

    def complete(self, global_request_id: int, worker_index: int) -> None:
        expected_worker = self.active.get(global_request_id)
        if expected_worker is None:
            if global_request_id in self.completed:
                raise RuntimeError(
                    f"duplicate completion for request {global_request_id}"
                )
            raise RuntimeError(
                f"completion for request {global_request_id} that is not active"
            )
        if worker_index != expected_worker:
            raise RuntimeError(
                f"request {global_request_id} completed by worker {worker_index}, "
                f"expected worker {expected_worker}"
            )
        del self.active[global_request_id]
        self.completed.add(global_request_id)

    def seconds_to_next_arrival(
        self, elapsed: float, schedule: Sequence[float]
    ) -> Optional[float]:
        if self.next_arrival >= self.total_requests:
            return None
        return max(0.0, schedule[self.next_arrival] - elapsed)

    @property
    def finished(self) -> bool:
        return len(self.completed) == self.total_requests


def split_integer(total: int, parts: int) -> List[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    if total < parts:
        raise ValueError(f"cannot split {total} into {parts} positive parts")
    quotient, remainder = divmod(total, parts)
    return [quotient + (index < remainder) for index in range(parts)]


def _strip_parent_options(command_args: Sequence[str]) -> List[str]:
    cleaned = []
    index = 0
    while index < len(command_args):
        token = command_args[index]
        if token in _PARENT_VALUE_OPTIONS:
            if index + 1 >= len(command_args):
                raise ValueError(f"{token} requires a value")
            index += 2
            continue
        if any(token.startswith(option + "=") for option in _PARENT_VALUE_OPTIONS):
            index += 1
            continue
        if token in _PARENT_FLAG_OPTIONS:
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return cleaned


def _build_worker_command(
    benchmark_args: Sequence[str],
    *,
    total_num_prompts: int,
    max_concurrency: Optional[int],
    request_rate: float,
    warmup_requests: int,
    result_file: Path,
    prepare_only: bool,
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        *_strip_parent_options(benchmark_args),
        "--num-prompts",
        str(total_num_prompts),
        "--disable-tqdm",
    ]
    if prepare_only:
        command.append("--prepare-only")
        return command

    command.extend(
        [
            "--request-rate",
            _format_rate(request_rate),
            "--warmup-requests",
            str(warmup_requests),
            "--output-file",
            str(result_file),
            "--output-details",
        ]
    )
    if max_concurrency is not None:
        command.extend(["--max-concurrency", str(max_concurrency)])
    return command


def _metric_stats(values: Sequence[float], name: str) -> Dict[str, float]:
    samples: Any = np.asarray(values, dtype=np.float64) if values else 0
    return {
        f"mean_{name}_ms": float(np.mean(samples) * 1000),
        f"median_{name}_ms": float(np.median(samples) * 1000),
        f"std_{name}_ms": float(np.std(samples) * 1000),
        f"p90_{name}_ms": float(np.percentile(samples, 90) * 1000),
        f"p95_{name}_ms": float(np.percentile(samples, 95) * 1000),
        f"p99_{name}_ms": float(np.percentile(samples, 99) * 1000),
    }


def _peak_metrics(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    peak_metrics = calculate_peak_metrics(
        [
            RequestTiming(
                start_time=float(sample["start_time"]),
                latency=float(sample["latency"]),
                ttft=float(sample["ttft"]),
                itls=sample["itls"],
            )
            for sample in samples
        ]
    )
    return {
        "max_output_tokens_per_s": peak_metrics.max_output_tokens_per_s,
        "max_concurrent_requests": peak_metrics.max_concurrent_requests,
    }


def _validate_detail_lengths(result: Dict[str, Any], worker_index: int) -> int:
    required = (*_DETAIL_KEYS, *_INTERNAL_DETAIL_KEYS, "global_request_ids")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(
            f"worker {worker_index} result lacks exact-aggregation fields: "
            f"{missing}. Run it with the matching serving.py version."
        )
    lengths = {key: len(result[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"worker {worker_index} detail arrays have different lengths: {lengths}"
        )
    count = next(iter(lengths.values()))
    for key in _OPTIONAL_DETAIL_KEYS:
        if key in result and len(result[key]) != count:
            raise ValueError(
                f"worker {worker_index} {key} has length "
                f"{len(result[key])}, expected {count}"
            )
    reported_completed = int(result.get("completed", sum(result["successes"])))
    observed_completed = sum(bool(success) for success in result["successes"])
    if reported_completed != observed_completed:
        raise ValueError(
            f"worker {worker_index} reports completed={reported_completed}, "
            f"but detail records contain {observed_completed} successes"
        )
    return count


def _merge_request_details(
    child_results: Sequence[Dict[str, Any]], total_num_prompts: int
) -> List[Dict[str, Any]]:
    for key in _OPTIONAL_DETAIL_KEYS:
        presence = [key in result for result in child_results]
        if any(presence) and not all(presence):
            raise ValueError(
                f"optional detail field {key} is present in only some workers"
            )

    records: Dict[int, Dict[str, Any]] = {}
    for worker_index, result in enumerate(child_results):
        count = _validate_detail_lengths(result, worker_index)
        available_optional = [key for key in _OPTIONAL_DETAIL_KEYS if key in result]
        for index in range(count):
            global_request_id = int(result["global_request_ids"][index])
            if global_request_id in records:
                raise ValueError(
                    f"duplicate global request id {global_request_id} in child results"
                )
            record = {
                key: result[key][index]
                for key in (
                    *_DETAIL_KEYS,
                    *_INTERNAL_DETAIL_KEYS,
                    *available_optional,
                )
            }
            record["global_request_id"] = global_request_id
            records[global_request_id] = record

    expected_ids = set(range(total_num_prompts))
    observed_ids = set(records)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        unexpected = sorted(observed_ids - expected_ids)
        raise ValueError(
            "child results do not cover the global request sequence: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return [records[index] for index in range(total_num_prompts)]


def aggregate_results(
    child_results: Sequence[Dict[str, Any]],
    total_num_prompts: int,
    total_concurrency: Optional[int],
    total_request_rate: float,
    *,
    include_details: bool = False,
) -> Dict[str, Any]:
    if not child_results:
        raise ValueError("no child benchmark results")

    benchmark_start_time = min(
        float(result["benchmark_start_time"]) for result in child_results
    )
    benchmark_end_time = max(
        float(result["benchmark_end_time"]) for result in child_results
    )
    duration = benchmark_end_time - benchmark_start_time
    if duration <= 0:
        raise ValueError(f"invalid aggregate benchmark duration: {duration}")

    request_records = _merge_request_details(child_results, total_num_prompts)
    last_result = max(
        child_results, key=lambda result: float(result["benchmark_end_time"])
    )
    use_retokenized_metric_itl = (
        last_result.get("accept_length") is not None
        and last_result["accept_length"] > 0
        and last_result.get("backend") in ("sglang-oai", "sglang-oai-chat")
    )
    successful_samples = []
    ttfts = []
    tpots = []
    metric_itls = []
    e2e_latencies = []
    for record in request_records:
        if not record["successes"]:
            continue
        output_len = int(record["output_lens"])
        latency = float(record["latencies"])
        ttft = float(record["ttfts"])
        raw_itls = record["itls"]
        request_metric_itls = record[
            (
                "_retokenized_metric_itls"
                if use_retokenized_metric_itl
                else "_raw_metric_itls"
            )
        ]
        start_time = float(record["start_times"])

        if output_len > 1:
            tpots.append((latency - ttft) / (output_len - 1))
        ttfts.append(ttft)
        metric_itls.extend(request_metric_itls)
        e2e_latencies.append(latency)
        successful_samples.append(
            {
                "start_time": start_time,
                "latency": latency,
                "ttft": ttft,
                "itls": raw_itls,
            }
        )

    completed = len(successful_samples)
    if completed == 0:
        raise ValueError("all requests failed; aggregate latency metrics are undefined")

    total_input = sum(int(result["total_input_tokens"]) for result in child_results)
    total_input_text = sum(
        int(result["total_input_text_tokens"]) for result in child_results
    )
    total_input_vision = sum(
        int(result["total_input_vision_tokens"]) for result in child_results
    )
    total_output = sum(int(result["total_output_tokens"]) for result in child_results)
    total_output_retokenized = sum(
        int(result["total_output_tokens_retokenized"]) for result in child_results
    )

    aggregate = {
        "tag": last_result.get("tag"),
        "backend": last_result.get("backend"),
        "dataset_name": last_result.get("dataset_name"),
        "request_rate": total_request_rate,
        "arrival_seed": last_result.get("arrival_seed"),
        "max_concurrency": total_concurrency,
        "sharegpt_output_len": last_result.get("sharegpt_output_len"),
        "random_input_len": last_result.get("random_input_len"),
        "random_output_len": last_result.get("random_output_len"),
        "random_range_ratio": last_result.get("random_range_ratio"),
        "num_prompts": total_num_prompts,
        "num_processes": len(child_results),
        "benchmark_start_time": benchmark_start_time,
        "benchmark_end_time": benchmark_end_time,
        "duration": duration,
        "completed": completed,
        "failed": total_num_prompts - completed,
        "total_input_tokens": total_input,
        "total_input_text_tokens": total_input_text,
        "total_input_vision_tokens": total_input_vision,
        "total_output_tokens": total_output,
        "total_output_tokens_retokenized": total_output_retokenized,
        "request_throughput": completed / duration,
        "input_throughput": total_input / duration,
        "output_throughput": total_output / duration,
        "output_throughput_retokenized": total_output_retokenized / duration,
        "total_throughput": (total_input + total_output) / duration,
        "total_throughput_retokenized": (total_input + total_output_retokenized)
        / duration,
        "concurrency": float(np.sum(e2e_latencies) / duration),
        "accept_length": last_result.get("accept_length"),
        "server_info": last_result.get("server_info"),
    }
    aggregate.update(_metric_stats(e2e_latencies, "e2e_latency"))
    aggregate.update(_metric_stats(ttfts, "ttft"))
    aggregate.update(_metric_stats(tpots, "tpot"))
    aggregate.update(_metric_stats(metric_itls, "itl"))
    aggregate["max_itl_ms"] = float(np.max(metric_itls or 0) * 1000)
    aggregate.update(_peak_metrics(successful_samples))

    cache_reports = [
        result.get("cache_report")
        for result in child_results
        if result.get("cache_report")
    ]
    if cache_reports:
        total_prompt_tokens = sum(
            int(report["total_prompt_tokens"]) for report in cache_reports
        )
        total_cached_tokens = sum(
            int(report["total_cached_tokens"]) for report in cache_reports
        )
        aggregate["cache_report"] = {
            "total_prompt_tokens": total_prompt_tokens,
            "total_cached_tokens": total_cached_tokens,
            "cache_hit_rate_pct": (
                round(100 * total_cached_tokens / total_prompt_tokens, 2)
                if total_prompt_tokens
                else 0.0
            ),
            "device_cached_tokens": _sum_optional(
                cache_reports, "device_cached_tokens"
            ),
            "host_cached_tokens": _sum_optional(cache_reports, "host_cached_tokens"),
            "storage_cached_tokens": _sum_optional(
                cache_reports, "storage_cached_tokens"
            ),
            "storage_backend": next(
                (
                    report.get("storage_backend")
                    for report in cache_reports
                    if report.get("storage_backend")
                ),
                None,
            ),
        }

    if include_details:
        for key in _DETAIL_KEYS:
            aggregate[key] = [record[key] for record in request_records]
        if use_retokenized_metric_itl:
            aggregate["metric_itls"] = [
                record["_retokenized_metric_itls"] for record in request_records
            ]
        for key in _OPTIONAL_DETAIL_KEYS:
            if all(key in record for record in request_records):
                aggregate[key] = [record[key] for record in request_records]

    return aggregate


def _sum_optional(reports: Sequence[Dict[str, Any]], key: str) -> Optional[int]:
    values = [report.get(key) for report in reports]
    if not any(value is not None for value in values):
        return None
    return sum(int(value or 0) for value in values)


def _format_rate(rate: float) -> str:
    return "inf" if math.isinf(rate) else repr(rate)


def _format_concurrency(max_concurrency: Optional[int]) -> str:
    return "not set" if max_concurrency is None else str(max_concurrency)


def print_aggregate_result(result: Dict[str, Any]) -> None:
    print("\n{s:{c}^{n}}".format(s=" Aggregate Serving Benchmark Result ", n=58, c="="))
    rows = (
        ("Backend:", result["backend"]),
        ("Processes:", result["num_processes"]),
        ("Traffic request rate:", result["request_rate"]),
        (
            "Max request concurrency:",
            _format_concurrency(result["max_concurrency"]),
        ),
        ("Successful requests:", result["completed"]),
        ("Failed requests:", result["failed"]),
        ("Benchmark duration (s):", f"{result['duration']:.2f}"),
        ("Total input tokens:", result["total_input_tokens"]),
        ("Total generated tokens:", result["total_output_tokens"]),
        ("Request throughput (req/s):", f"{result['request_throughput']:.2f}"),
        ("Input token throughput (tok/s):", f"{result['input_throughput']:.2f}"),
        ("Output token throughput (tok/s):", f"{result['output_throughput']:.2f}"),
        (
            "Peak output token throughput (tok/s):",
            f"{result['max_output_tokens_per_s']:.2f}",
        ),
        ("Peak concurrent requests:", result["max_concurrent_requests"]),
        ("Total token throughput (tok/s):", f"{result['total_throughput']:.2f}"),
        ("Concurrency:", f"{result['concurrency']:.2f}"),
    )
    for label, value in rows:
        print(f"{label:<42} {value}")

    for title, key in (
        ("End-to-End Latency", "e2e_latency"),
        ("Time to First Token", "ttft"),
        ("Time per Output Token", "tpot"),
        ("Inter-Token Latency", "itl"),
    ):
        print(f"{title:-^58}")
        for percentile, label in (
            ("mean", "Mean"),
            ("median", "Median"),
            ("p90", "P90"),
            ("p95", "P95"),
            ("p99", "P99"),
        ):
            value = result[f"{percentile}_{key}_ms"]
            print(f"{label + ' ' + title + ' (ms):':<42} {value:.2f}")
    print("=" * 58)


def _resolve_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    default_ports = {
        "sglang": 30000,
        "sglang-native": 30000,
        "sglang-oai": 30000,
        "sglang-oai-chat": 30000,
        "sglang-embedding": 30000,
        "lmdeploy": 23333,
        "lmdeploy-chat": 23333,
        "vllm": 8000,
        "vllm-chat": 8000,
        "vllm-embedding": 8000,
        "trt": 8000,
        "gserver": 9988,
        "truss": 8080,
    }
    port = args.port or default_ports.get(args.backend, 30000)
    host = args.host
    formatted_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{formatted_host}:{port}"


def _flush_cache(base_url: str, backend: str) -> None:
    headers = {}
    if api_key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {api_key}"
    elif api_key := os.environ.get("API_KEY"):
        headers["Authorization"] = api_key

    cache_endpoint = (
        "/reset_prefix_cache" if backend.startswith("vllm") else "/flush_cache"
    )
    request = urllib.request.Request(
        base_url.rstrip("/") + cache_endpoint,
        data=b"",
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"flush_cache returned HTTP {response.status}")


def _tail(path: Path, line_count: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-line_count:])
    except OSError:
        return "<unable to read worker log>"


def _check_worker_failures(workers: Sequence[Worker]) -> None:
    for worker in workers:
        return_code = worker.process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"worker {worker.index} exited before coordination completed "
                f"with code {return_code}:\n"
                f"{_tail(worker.log_file)}"
            )


def _read_result(path: Path) -> Dict[str, Any]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one JSON result in {path}, found {len(lines)}")
    return json.loads(lines[0])


def _terminate_workers(workers: Sequence[Worker]) -> None:
    for worker in workers:
        if worker.process.poll() is None:
            worker.process.terminate()
    for worker in workers:
        try:
            worker.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.process.kill()
            worker.process.wait()


async def _wait_for_workers(workers: Sequence[Worker]) -> None:
    unfinished = set(range(len(workers)))
    while unfinished:
        for worker_index in list(unfinished):
            worker = workers[worker_index]
            return_code = worker.process.poll()
            if return_code is None:
                continue
            unfinished.remove(worker_index)
            worker.log_handle.close()
            if return_code != 0:
                raise RuntimeError(
                    f"worker {worker.index} failed with code {return_code}:\n"
                    f"{_tail(worker.log_file)}"
                )
        if unfinished:
            await asyncio.sleep(0.05)


def _decode_worker_message(
    frames: Sequence[bytes], run_id: str
) -> tuple[int, Dict[str, Any]]:
    if len(frames) != 2:
        raise RuntimeError(f"invalid ZMQ control message with {len(frames)} frames")
    identity, payload = frames
    try:
        worker_index = int(identity.decode("ascii"))
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid ZMQ control message") from exc
    if message.get("run_id") != run_id:
        raise RuntimeError(
            f"control run id mismatch: {message.get('run_id')} != {run_id}"
        )
    if int(message.get("worker_index", -1)) != worker_index:
        raise RuntimeError(
            f"worker identity mismatch: frame={worker_index}, "
            f"payload={message.get('worker_index')}"
        )
    return worker_index, message


async def _send_worker_message(
    socket: zmq.asyncio.Socket,
    worker_index: int,
    run_id: str,
    message: Dict[str, Any],
) -> None:
    payload = dict(message)
    payload["run_id"] = run_id
    await socket.send_multipart(
        [
            str(worker_index).encode("ascii"),
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        ]
    )


async def _wait_for_ready(
    socket: zmq.asyncio.Socket,
    workers: Sequence[Worker],
    run_id: str,
    timeout_s: float,
) -> None:
    expected_counts = {worker.index: worker.num_prompts for worker in workers}
    ready = set()
    deadline = time.monotonic() + timeout_s
    while len(ready) < len(workers):
        _check_worker_failures(workers)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missing = sorted(set(expected_counts) - ready)
            raise TimeoutError(
                f"workers {', '.join(map(str, missing))} did not reach "
                "the start barrier"
            )
        if not await socket.poll(timeout=max(1, int(min(remaining, 0.25) * 1000))):
            continue
        worker_index, message = _decode_worker_message(
            await socket.recv_multipart(), run_id
        )
        if message.get("type") != "ready":
            raise RuntimeError(
                f"unexpected worker {worker_index} message before start: {message}"
            )
        if worker_index in ready:
            raise RuntimeError(f"worker {worker_index} sent ready twice")
        request_count = int(message.get("request_count", -1))
        if request_count != expected_counts.get(worker_index):
            raise RuntimeError(
                f"worker {worker_index} prepared {request_count} requests, "
                f"expected {expected_counts.get(worker_index)}"
            )
        ready.add(worker_index)


async def _run_global_schedule(
    socket: zmq.asyncio.Socket,
    workers: Sequence[Worker],
    run_id: str,
    schedule: Sequence[float],
    start_time: float,
    max_concurrency: Optional[int],
) -> None:
    state = GlobalAdmissionState(len(schedule), max_concurrency)
    worker_count = len(workers)
    if start_time > time.perf_counter():
        await asyncio.sleep(start_time - time.perf_counter())

    while not state.finished:
        _check_worker_failures(workers)
        elapsed = time.perf_counter() - start_time
        state.enqueue_due(elapsed, schedule)
        for global_id, worker_index, local_id in state.take_dispatchable(worker_count):
            await _send_worker_message(
                socket,
                worker_index,
                run_id,
                {
                    "type": "dispatch",
                    "global_request_id": global_id,
                    "local_request_id": local_id,
                },
            )

        if state.finished:
            break

        next_arrival_delay = state.seconds_to_next_arrival(elapsed, schedule)
        poll_seconds = (
            0.25 if next_arrival_delay is None else min(0.25, next_arrival_delay)
        )
        if poll_seconds <= 0:
            continue
        if not await socket.poll(timeout=max(1, math.ceil(poll_seconds * 1000))):
            continue

        worker_index, message = _decode_worker_message(
            await socket.recv_multipart(), run_id
        )
        message_type = message.get("type")
        global_request_id = int(message.get("global_request_id", -1))
        if message_type == "done":
            state.complete(global_request_id, worker_index)
        elif message_type == "error":
            raise RuntimeError(
                f"worker {worker_index} request {global_request_id} failed in "
                f"the load generator: {message.get('error')}"
            )
        else:
            raise RuntimeError(
                f"unexpected worker {worker_index} message during benchmark: "
                f"{message}"
            )


async def _abort_workers(
    socket: zmq.asyncio.Socket,
    workers: Sequence[Worker],
    run_id: str,
    error: str,
) -> None:
    for worker in workers:
        if worker.process.poll() is None:
            try:
                await _send_worker_message(
                    socket,
                    worker.index,
                    run_id,
                    {"type": "abort", "error": error},
                )
            except zmq.ZMQError:
                pass


def _worker_summary(result: Dict[str, Any], worker: Worker) -> Dict[str, Any]:
    summary = {
        "worker": worker.index,
        "assigned_num_prompts": worker.num_prompts,
        "result_file": str(worker.result_file),
        "log_file": str(worker.log_file),
    }
    metric_keys = (
        "duration",
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "total_throughput",
        "concurrency",
        "max_output_tokens_per_s",
        "max_concurrent_requests",
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
        "std_e2e_latency_ms",
        "p90_e2e_latency_ms",
        "p95_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "mean_ttft_ms",
        "median_ttft_ms",
        "std_ttft_ms",
        "p90_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p90_tpot_ms",
        "p95_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "std_itl_ms",
        "p90_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
    )
    summary.update({key: result.get(key) for key in metric_keys})
    return summary


def _default_output_file(args: argparse.Namespace) -> str:
    if args.output_file:
        return args.output_file
    now = datetime.now().strftime("%m%d")
    if args.dataset_name == "image":
        return (
            f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_"
            f"{args.random_output_len}_{args.image_count}imgs_"
            f"{args.image_resolution}.jsonl"
        )
    if args.dataset_name.startswith("random"):
        return (
            f"{args.backend}_{now}_{args.num_prompts}_{args.random_input_len}_"
            f"{args.random_output_len}.jsonl"
        )
    return f"{args.backend}_{now}_{args.num_prompts}_{args.dataset_name}.jsonl"


def _validate_multiprocess_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.num_processes <= 0:
        parser.error("--num-processes must be positive")
    if args.num_prompts < args.num_processes:
        parser.error("--num-prompts must be at least --num-processes")
    if args.max_concurrency is not None and args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive")
    if args.multiprocess_barrier_timeout_sec <= 0:
        parser.error("--multiprocess-barrier-timeout-sec must be positive")
    if not math.isinf(args.request_rate) and (
        not math.isfinite(args.request_rate) or args.request_rate <= 0
    ):
        parser.error("--request-rate must be positive or inf")
    if args.dataset_name == "mooncake" or args.use_trace_timestamps:
        parser.error(
            "multi-process benchmark does not support Mooncake trace scheduling"
        )
    if args.profile:
        parser.error("multi-process benchmark does not support --profile")
    if args.plot_throughput:
        parser.error("multi-process benchmark does not support --plot-throughput")
    if args.lora_name:
        parser.error(
            "multi-process benchmark does not yet support LoRA request "
            "assignment with exact single-process ordering"
        )
    if args.dataset_name == "agentic-trace" or (
        args.dataset_name == "generated-shared-prefix" and args.gsp_num_turns != 1
    ):
        parser.error("multi-process benchmark does not yet support multi-turn requests")
    if args.dataset_name == "generated-shared-prefix":
        generated_count = args.gsp_num_groups * args.gsp_prompts_per_group
        if generated_count != args.num_prompts:
            parser.error(
                "generated-shared-prefix ignores --num-prompts: "
                f"--gsp-num-groups ({args.gsp_num_groups}) * "
                f"--gsp-prompts-per-group ({args.gsp_prompts_per_group}) = "
                f"{generated_count}, but --num-prompts is {args.num_prompts}"
            )


def _start_workers(
    args: argparse.Namespace,
    benchmark_args: Sequence[str],
    child_output_dir: Path,
    *,
    endpoint: Optional[str],
    run_id: str,
) -> List[Worker]:
    prompt_counts = split_integer(args.num_prompts, args.num_processes)
    gsp_run_id = uuid.uuid4().hex[:8]
    gsp_run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    workers = []
    for index, prompt_count in enumerate(prompt_counts):
        result_file = child_output_dir / f"worker_{index:02d}.jsonl"
        log_file = child_output_dir / f"worker_{index:02d}.log"
        command = _build_worker_command(
            benchmark_args,
            total_num_prompts=args.num_prompts,
            max_concurrency=args.max_concurrency,
            request_rate=args.request_rate,
            warmup_requests=args.warmup_requests if index == 0 else 0,
            result_file=result_file,
            prepare_only=args.prepare_only,
        )
        environment = os.environ.copy()
        for name in (
            CONTROL_ENDPOINT_ENV,
            CONTROL_RUN_ID_ENV,
            CONTROL_WORKER_INDEX_ENV,
        ):
            environment.pop(name, None)
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "SGLANG_BENCH_NUM_SHARDS": str(args.num_processes),
                "SGLANG_BENCH_SHARD_INDEX": str(index),
                "SGLANG_BENCH_EXPECTED_NUM_REQUESTS": str(args.num_prompts),
                "SGLANG_BENCH_GSP_RUN_ID": gsp_run_id,
                "SGLANG_BENCH_GSP_RUN_TIMESTAMP": gsp_run_timestamp,
            }
        )
        if not args.output_details:
            environment["SGLANG_BENCH_OMIT_GENERATED_TEXT"] = "1"
        else:
            environment.pop("SGLANG_BENCH_OMIT_GENERATED_TEXT", None)
        if endpoint is not None:
            environment.update(
                {
                    CONTROL_ENDPOINT_ENV: endpoint,
                    CONTROL_RUN_ID_ENV: run_id,
                    CONTROL_WORKER_INDEX_ENV: str(index),
                }
            )

        log_handle = log_file.open("w")
        try:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        except Exception:
            log_handle.close()
            raise
        workers.append(
            Worker(
                index=index,
                num_prompts=prompt_count,
                process=process,
                log_file=log_file,
                result_file=result_file,
                log_handle=log_handle,
            )
        )
    return workers


async def _run_multiprocess_benchmark_async(
    args: argparse.Namespace,
    benchmark_args: Sequence[str],
    child_output_dir: Path,
    schedule: Sequence[float],
    schedule_file: Path,
) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex
    socket_path = (
        Path(tempfile.gettempdir()) / f"sgbench-{os.getpid()}-{run_id[:8]}.sock"
    )
    endpoint = f"ipc://{socket_path}"
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.ROUTER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
    socket.bind(endpoint)
    workers: List[Worker] = []
    try:
        workers = _start_workers(
            args,
            benchmark_args,
            child_output_dir,
            endpoint=endpoint,
            run_id=run_id,
        )
        await _wait_for_ready(
            socket,
            workers,
            run_id,
            args.multiprocess_barrier_timeout_sec,
        )
        should_flush_cache = args.flush_cache or (
            "sglang" in args.backend
            and os.getenv("SGLANG_IS_IN_CI", "").lower() in ("1", "true")
        )
        if should_flush_cache:
            _flush_cache(_resolve_base_url(args), args.backend)

        benchmark_start_time = time.perf_counter() + 1.0
        for worker in workers:
            await _send_worker_message(
                socket,
                worker.index,
                run_id,
                {"type": "start", "start_time": benchmark_start_time},
            )
        await _run_global_schedule(
            socket,
            workers,
            run_id,
            schedule,
            benchmark_start_time,
            args.max_concurrency,
        )
        for worker in workers:
            await _send_worker_message(
                socket,
                worker.index,
                run_id,
                {"type": "finish"},
            )
        await _wait_for_workers(workers)

        child_results = [_read_result(worker.result_file) for worker in workers]
        aggregate = aggregate_results(
            child_results=child_results,
            total_num_prompts=args.num_prompts,
            total_concurrency=args.max_concurrency,
            total_request_rate=args.request_rate,
            include_details=args.output_details,
        )
        aggregate["workers"] = [
            _worker_summary(result, worker)
            for result, worker in zip(child_results, workers)
        ]
        aggregate["child_output_dir"] = str(child_output_dir.resolve())
        aggregate["arrival_seed"] = (
            args.seed if args.arrival_seed is None else args.arrival_seed
        )
        aggregate["arrival_schedule_file"] = str(schedule_file.resolve())
        return aggregate
    except BaseException as exc:
        if workers:
            await _abort_workers(socket, workers, run_id, str(exc))
            _terminate_workers(workers)
        raise
    finally:
        for worker in workers:
            if not worker.log_handle.closed:
                worker.log_handle.close()
        socket.close(linger=0)
        context.term()
        socket_path.unlink(missing_ok=True)


async def _run_prepare_only_async(
    args: argparse.Namespace,
    benchmark_args: Sequence[str],
    child_output_dir: Path,
) -> None:
    workers: List[Worker] = []
    try:
        workers = _start_workers(
            args,
            benchmark_args,
            child_output_dir,
            endpoint=None,
            run_id="",
        )
        await _wait_for_workers(workers)
    except BaseException:
        _terminate_workers(workers)
        raise
    finally:
        for worker in workers:
            if not worker.log_handle.closed:
                worker.log_handle.close()


def run_multiprocess_benchmark(
    args: argparse.Namespace,
    benchmark_args: Sequence[str],
    parser: argparse.ArgumentParser,
) -> Dict[str, Any]:
    """Run a coordinated benchmark from the normal bench_serving CLI."""

    _validate_multiprocess_args(parser, args)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    child_output_dir = (
        Path(args.multiprocess_child_output_dir)
        if args.multiprocess_child_output_dir
        else Path(f"multiprocess_benchmark_{timestamp}_{os.getpid()}")
    )
    child_output_dir.mkdir(parents=True, exist_ok=False)

    if args.prepare_only:
        asyncio.run(_run_prepare_only_async(args, benchmark_args, child_output_dir))
        print(
            "\nDataset preparation completed for "
            f"{args.num_prompts} requests across {args.num_processes} processes."
        )
        print(f"Worker logs: {child_output_dir.resolve()}")
        return {"prepared_requests": args.num_prompts}

    arrival_seed = args.seed if args.arrival_seed is None else args.arrival_seed
    schedule = generate_arrival_schedule(
        args.num_prompts, args.request_rate, arrival_seed
    )
    schedule_file = child_output_dir / "arrival_schedule.json"
    schedule_file.write_text(json.dumps(schedule))
    print(
        "Global arrival schedule: "
        f"seed={arrival_seed}, requests={len(schedule)}, "
        f"span={schedule[-1]:.6f}s"
    )
    print(
        "Global concurrency coordinator: "
        f"max_concurrency={_format_concurrency(args.max_concurrency)}"
    )

    aggregate = asyncio.run(
        _run_multiprocess_benchmark_async(
            args,
            benchmark_args,
            child_output_dir,
            schedule,
            schedule_file,
        )
    )
    print("\nPer-process result:")
    for worker_result in aggregate["workers"]:
        print(
            f"  worker {worker_result['worker']}: "
            f"completed={worker_result['completed']}, "
            f"req/s={worker_result['request_throughput']:.2f}, "
            f"output tok/s={worker_result['output_throughput']:.2f}, "
            f"TTFT={worker_result['mean_ttft_ms']:.2f} ms, "
            f"TPOT={worker_result['mean_tpot_ms']:.2f} ms"
        )
    print_aggregate_result(aggregate)

    output_file = Path(_default_output_file(args))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a") as file:
        file.write(json.dumps(aggregate) + "\n")
    print(f"Aggregate result: {output_file.resolve()}")
    print(f"Child results and logs: {child_output_dir.resolve()}")
    return aggregate
