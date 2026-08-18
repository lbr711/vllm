# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from unittest.mock import Mock, patch

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
