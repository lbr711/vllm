# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import patch

from vllm.entrypoints.serve.snapshot.sentinel import (
    DEVICE_UNLOCK_TIMEOUT,
    RESUME_TIMEOUT,
    SUSPEND_TIMEOUT,
    SnapshotSentinel,
)


def _sentinel(metadata_path: str) -> SnapshotSentinel:
    return SnapshotSentinel(
        snapshot_metadata=metadata_path,
        host="127.0.0.1",
        port=8000,
        use_tls=False,
        ca_file=None,
    )


def test_suspend_uses_snapshot_metadata(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"model_save_path": "/snapshot/weights"}),
        encoding="utf-8",
    )
    sentinel = _sentinel(str(metadata_path))

    with patch.object(sentinel, "_request") as request:
        sentinel._call_suspend()

    request.assert_called_once_with(
        "POST",
        "/suspend",
        SUSPEND_TIMEOUT,
        {"model_save_path": "/snapshot/weights"},
    )


def test_checkpoint_unlocks_device_and_stops_on_cold_start(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({"checkpoint": "done"}), encoding="utf-8")
    sentinel = _sentinel(str(metadata_path))

    with (
        patch.object(sentinel, "_request") as request,
        patch(
            "vllm.entrypoints.serve.snapshot.sentinel."
            "is_restored_from_host_side_snapshot",
            return_value=False,
        ),
    ):
        sentinel._reach_checkpoint()

    request.assert_called_once_with("POST", "/device_unlock", DEVICE_UNLOCK_TIMEOUT)
    assert sentinel._stop_event.is_set()


def test_resume_uses_snapshot_metadata(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_load_path": "/snapshot/weights",
                "data_parallel_master_ip": "10.0.0.1",
            }
        ),
        encoding="utf-8",
    )
    sentinel = _sentinel(str(metadata_path))

    with patch.object(sentinel, "_request") as request:
        sentinel._call_resume()

    request.assert_called_once_with(
        "POST",
        "/resume",
        RESUME_TIMEOUT,
        {
            "model_path": "/snapshot/weights",
            "data_parallel_master_ip": "10.0.0.1",
        },
    )
