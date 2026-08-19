# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm.entrypoints.serve.snapshot.utils import RETRY_INTERVAL
from vllm.v1.engine.core import EngineCoreProc


def test_transport_restore_retries_until_master_ip_is_available():
    engine_core = object.__new__(EngineCoreProc)
    engine_core._transport_lock = threading.Lock()
    engine_core._transport_restored = False
    engine_core._reconnect_transport = Mock()

    with (
        patch("vllm.v1.engine.core.is_restore", return_value=True),
        patch(
            "vllm.entrypoints.serve.snapshot.utils.load_snapshot_metadata",
            side_effect=[ValueError("field is not ready"), "10.0.0.2"],
        ) as load_metadata,
        patch("vllm.v1.engine.core.time.sleep") as sleep,
    ):
        engine_core._restore_transport_from_metadata("/snapshot/metadata.json")

    assert load_metadata.call_count == 2
    sleep.assert_called_once_with(RETRY_INTERVAL)
    engine_core._reconnect_transport.assert_called_once_with("10.0.0.2")
    assert engine_core._transport_restored


def test_disconnected_input_transport_stops_without_lingering():
    engine_core = object.__new__(EngineCoreProc)
    engine_core.tensor_ipc_receiver = None
    engine_core.frontend_stats_publish_address = None
    engine_core.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1, dtype=torch.float16),
        cache_config=SimpleNamespace(num_gpu_blocks=1, block_size=1),
    )
    ready_event = threading.Event()
    stop_event = threading.Event()
    stop_reader, stop_writer = os.pipe()
    input_thread = threading.Thread(
        target=engine_core.process_input_sockets,
        args=(
            ["tcp://127.0.0.1:1"],
            None,
            b"engine-0",
            ready_event,
            stop_event,
            stop_reader,
        ),
        daemon=True,
    )

    input_thread.start()
    assert ready_event.wait(timeout=1)
    stop_event.set()
    os.write(stop_writer, b"\0")
    input_thread.join(timeout=1)
    os.close(stop_reader)
    os.close(stop_writer)

    assert not input_thread.is_alive()
