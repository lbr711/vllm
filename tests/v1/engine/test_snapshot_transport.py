# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
from unittest.mock import Mock, patch

import zmq

from vllm.snapshot.utils import RETRY_INTERVAL
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import EngineZmqAddresses


def test_transport_reconnect_is_not_required_for_ipc_only():
    addresses = EngineZmqAddresses(
        inputs=["ipc:///tmp/input"],
        outputs=["ipc:///tmp/output"],
    )

    assert not EngineCoreProc._transport_requires_reconnect(addresses)


def test_transport_reconnect_is_required_for_tcp_coordinator():
    addresses = EngineZmqAddresses(
        inputs=["ipc:///tmp/input"],
        outputs=["ipc:///tmp/output"],
        coordinator_input="tcp://10.0.0.1:1234",
    )

    assert EngineCoreProc._transport_requires_reconnect(addresses)


def test_transport_restore_retries_until_master_ip_is_available():
    engine_core = object.__new__(EngineCoreProc)
    engine_core._transport_lock = threading.Lock()
    engine_core._transport_restored = False
    engine_core._reconnect_transport = Mock()

    with (
        patch("vllm.v1.engine.core.is_restore", return_value=True),
        patch(
            "vllm.snapshot.utils.load_snapshot_metadata",
            side_effect=[ValueError("field is not ready"), "10.0.0.2"],
        ) as load_metadata,
        patch("vllm.v1.engine.core.time.sleep") as sleep,
    ):
        engine_core._restore_transport_from_metadata("/snapshot/metadata.json")

    assert load_metadata.call_count == 2
    sleep.assert_called_once_with(RETRY_INTERVAL)
    engine_core._reconnect_transport.assert_called_once_with("10.0.0.2")
    assert engine_core._transport_restored


def test_stop_pipe_wakes_zmq_poller():
    stop_reader, stop_writer = os.pipe()
    stopped = threading.Event()

    def poll_stop_pipe():
        poller = zmq.Poller()
        poller.register(stop_reader, zmq.POLLIN)
        events = poller.poll(timeout=1000)
        if events and events[0][0] == stop_reader:
            os.read(stop_reader, 1)
            stopped.set()

    input_thread = threading.Thread(target=poll_stop_pipe, daemon=True)
    input_thread.start()
    os.write(stop_writer, b"\0")
    input_thread.join(timeout=1)
    os.close(stop_reader)
    os.close(stop_writer)

    assert not input_thread.is_alive()
    assert stopped.is_set()
