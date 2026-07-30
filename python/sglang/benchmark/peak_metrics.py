# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np


class RequestTiming(NamedTuple):
    start_time: float
    latency: float
    ttft: float
    itls: Sequence[float]


@dataclass
class PeakMetrics:
    max_output_tokens_per_s: float
    max_concurrent_requests: int
    tokens_per_second: np.ndarray
    concurrent_requests_per_second: np.ndarray


def calculate_peak_metrics(samples: Sequence[RequestTiming]) -> PeakMetrics:
    if not samples:
        return PeakMetrics(
            max_output_tokens_per_s=0.0,
            max_concurrent_requests=0,
            tokens_per_second=np.zeros(0),
            concurrent_requests_per_second=np.zeros(0),
        )

    min_start_time = min(sample.start_time for sample in samples)
    max_end_time = max(sample.start_time + sample.latency for sample in samples)
    duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
    tokens_per_second = np.zeros(duration_seconds)
    concurrent_requests_per_second = np.zeros(duration_seconds)

    for sample in samples:
        token_times = [sample.start_time + sample.ttft]
        current_time = token_times[0]
        for itl_value in sample.itls:
            current_time += itl_value
            token_times.append(current_time)

        for token_time in token_times:
            second_bucket = int(token_time - min_start_time)
            if 0 <= second_bucket < duration_seconds:
                tokens_per_second[second_bucket] += 1

        request_start_second = int(sample.start_time - min_start_time)
        request_end_second = int(sample.start_time + sample.latency - min_start_time)
        for second in range(
            request_start_second, min(request_end_second + 1, duration_seconds)
        ):
            concurrent_requests_per_second[second] += 1

    return PeakMetrics(
        max_output_tokens_per_s=float(np.max(tokens_per_second)),
        max_concurrent_requests=int(np.max(concurrent_requests_per_second)),
        tokens_per_second=tokens_per_second,
        concurrent_requests_per_second=concurrent_requests_per_second,
    )
