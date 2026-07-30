# SPDX-License-Identifier: Apache-2.0

import math
from typing import List

import numpy as np


def generate_arrival_schedule(
    num_requests: int, request_rate: float, seed: int
) -> List[float]:
    """Generate absolute request offsets for an open-loop benchmark."""
    if num_requests < 0:
        raise ValueError("num_requests must be non-negative")
    if num_requests == 0:
        return []
    if math.isinf(request_rate):
        return [0.0] * num_requests
    if not math.isfinite(request_rate) or request_rate <= 0:
        raise ValueError("request_rate must be positive or inf")

    offsets = np.zeros(num_requests, dtype=np.float64)
    if num_requests > 1:
        rng = np.random.default_rng(seed)
        intervals = rng.exponential(1.0 / request_rate, num_requests - 1)
        offsets[1:] = np.cumsum(intervals)
    return offsets.tolist()
