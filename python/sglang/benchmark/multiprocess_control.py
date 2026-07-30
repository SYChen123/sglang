# SPDX-License-Identifier: Apache-2.0

"""ZMQ control plane for multi-process serving benchmarks."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import zmq
import zmq.asyncio

CONTROL_ENDPOINT_ENV = "SGLANG_BENCH_CONTROL_ENDPOINT"
CONTROL_RUN_ID_ENV = "SGLANG_BENCH_CONTROL_RUN_ID"
CONTROL_WORKER_INDEX_ENV = "SGLANG_BENCH_CONTROL_WORKER_INDEX"


@dataclass(frozen=True)
class Dispatch:
    global_request_id: int
    local_request_id: int


def is_controlled_worker() -> bool:
    return CONTROL_ENDPOINT_ENV in os.environ


class MultiprocessWorkerControl:
    """Worker-side endpoint for coordinator-driven request admission."""

    def __init__(self) -> None:
        endpoint = os.environ[CONTROL_ENDPOINT_ENV]
        self.run_id = os.environ[CONTROL_RUN_ID_ENV]
        self.worker_index = int(os.environ[CONTROL_WORKER_INDEX_ENV])
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, str(self.worker_index).encode("ascii"))
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IMMEDIATE, 1)
        self._socket.connect(endpoint)
        self._send_lock = asyncio.Lock()

    async def _send(self, message: dict) -> None:
        message["run_id"] = self.run_id
        message["worker_index"] = self.worker_index
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        async with self._send_lock:
            await self._socket.send(payload)

    async def ready_and_wait_start(self, request_count: int) -> float:
        await self._send({"type": "ready", "request_count": request_count})
        while True:
            message = json.loads((await self._socket.recv()).decode("utf-8"))
            self._validate_message(message)
            if message.get("type") == "start":
                return float(message["start_time"])
            if message.get("type") == "abort":
                raise RuntimeError(message.get("error", "benchmark aborted"))
            raise RuntimeError(f"unexpected coordinator message: {message}")

    async def iter_dispatches(self, expected_count: int) -> AsyncIterator[Dispatch]:
        for _ in range(expected_count):
            message = json.loads((await self._socket.recv()).decode("utf-8"))
            self._validate_message(message)
            message_type = message.get("type")
            if message_type == "abort":
                raise RuntimeError(message.get("error", "benchmark aborted"))
            if message_type != "dispatch":
                raise RuntimeError(f"unexpected coordinator message: {message}")
            yield Dispatch(
                global_request_id=int(message["global_request_id"]),
                local_request_id=int(message["local_request_id"]),
            )

    async def request_done(self, global_request_id: int, success: bool) -> None:
        await self._send(
            {
                "type": "done",
                "global_request_id": global_request_id,
                "success": success,
            }
        )

    async def request_error(self, global_request_id: int, error: BaseException) -> None:
        await self._send(
            {
                "type": "error",
                "global_request_id": global_request_id,
                "error": repr(error),
            }
        )

    async def wait_for_finish(self) -> None:
        message = json.loads((await self._socket.recv()).decode("utf-8"))
        self._validate_message(message)
        if message.get("type") == "finish":
            return
        if message.get("type") == "abort":
            raise RuntimeError(message.get("error", "benchmark aborted"))
        raise RuntimeError(f"unexpected coordinator message: {message}")

    def _validate_message(self, message: dict) -> None:
        if message.get("run_id") != self.run_id:
            raise RuntimeError(
                f"control run id mismatch: {message.get('run_id')} != {self.run_id}"
            )

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()


def create_worker_control() -> Optional[MultiprocessWorkerControl]:
    if not is_controlled_worker():
        return None
    required = (
        CONTROL_ENDPOINT_ENV,
        CONTROL_RUN_ID_ENV,
        CONTROL_WORKER_INDEX_ENV,
    )
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"incomplete multi-process control environment: {', '.join(missing)}"
        )
    return MultiprocessWorkerControl()
