# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from vllm.snapshot.lifecycle import resume_engine, suspend_engine, unlock_engine


def _engine(*, transport_restored: bool, dp_group=None):
    parallel_config = SimpleNamespace(
        data_parallel_master_ip="10.0.0.1",
        _data_parallel_master_port_list=[1234],
        stateless_init_dp_group=Mock(return_value="new-dp-group"),
    )
    engine = SimpleNamespace(
        _transport_lock=nullcontext(),
        _transport_restored=transport_restored,
        _reconnect_transport=Mock(),
        collective_rpc=Mock(),
        dp_group=dp_group,
        vllm_config=SimpleNamespace(
            parallel_config=parallel_config,
            kv_transfer_config=None,
        ),
    )
    return engine


def test_suspend_and_unlock_preserve_collective_rpc_order():
    engine = _engine(transport_restored=True)

    with patch("vllm.snapshot.lifecycle.gc.collect") as collect:
        suspend_engine(engine, "/snapshot/model")
    unlock_engine(engine)

    collect.assert_called_once_with()
    assert engine.collective_rpc.call_args_list == [
        call("dump_model", args=("/snapshot/model",)),
        call("aclrt_snapshot_process_lock"),
        call("aclrt_snapshot_process_backup"),
        call("aclrt_snapshot_process_unlock"),
    ]


def test_resume_reconnects_transport_before_worker_restore():
    engine = _engine(transport_restored=False)

    with (
        patch("vllm.snapshot.lifecycle.get_local_ip", return_value="10.0.0.2"),
        patch("vllm.snapshot.lifecycle.refresh_scheduler_after_resume") as refresh,
        patch(
            "vllm.snapshot.lifecycle.refresh_scheduler_handshake_metadata_after_resume"
        ) as refresh_metadata,
        patch.dict(os.environ, {}, clear=True),
    ):
        resume_engine(engine, "10.0.0.3", "/snapshot/model")

        assert engine.vllm_config.parallel_config.data_parallel_master_ip == "10.0.0.3"
        assert engine.collective_rpc.call_args_list == [
            call("aclrt_snapshot_process_restore"),
            call("aclrt_snapshot_process_unlock"),
            call(
                "update_worker_info_after_resume",
                args=("10.0.0.2", "10.0.0.3"),
            ),
            call("rebuild_parallel_group_after_resume"),
            call("re_load_weights", args=("/snapshot/model",)),
            call("recapture_graph"),
            call(
                "rebuild_kv_transfer_engine_after_resume",
                args=("10.0.0.2", None),
            ),
        ]
        assert os.environ["HCCL_IF_IP"] == "10.0.0.2"

    engine._reconnect_transport.assert_called_once_with("10.0.0.3")
    refresh.assert_called_once_with(engine, "10.0.0.2")
    refresh_metadata.assert_called_once_with(engine)


def test_resume_rebuilds_engine_core_dp_group():
    engine = _engine(transport_restored=True, dp_group="old-dp-group")

    with (
        patch("vllm.snapshot.lifecycle.get_local_ip", return_value="10.0.0.2"),
        patch(
            "vllm.snapshot.lifecycle.stateless_destroy_torch_distributed_process_group"
        ) as destroy_dp_group,
        patch("vllm.snapshot.lifecycle.refresh_scheduler_after_resume"),
        patch(
            "vllm.snapshot.lifecycle.refresh_scheduler_handshake_metadata_after_resume"
        ),
    ):
        resume_engine(engine, "10.0.0.3", None)

    destroy_dp_group.assert_called_once_with("old-dp-group")
    assert engine.vllm_config.parallel_config._data_parallel_master_port_list == []
    assert engine.dp_group == "new-dp-group"
