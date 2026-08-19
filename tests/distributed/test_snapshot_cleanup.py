# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.snapshot.distributed import cleanup_dist_env_for_snapshot


def _group_coordinator(device_group: Mock) -> GroupCoordinator:
    coordinator = object.__new__(GroupCoordinator)
    coordinator.device_group = device_group
    coordinator.device_communicator = None
    coordinator.mq_broadcaster = None
    return coordinator


def test_group_coordinator_uses_default_process_group_destroyer():
    device_group = Mock()
    coordinator = _group_coordinator(device_group)

    with patch("torch.distributed.destroy_process_group") as destroy:
        coordinator.destroy()

    destroy.assert_called_once_with(device_group)


def test_group_coordinator_accepts_snapshot_process_group_destroyer():
    device_group = Mock()
    coordinator = _group_coordinator(device_group)
    destroy = Mock()

    coordinator.destroy(destroy)

    destroy.assert_called_once_with(device_group)


def test_snapshot_cleanup_uses_hccl_abort_destroyer():
    with (
        patch("vllm.snapshot.distributed.destroy_model_parallel") as destroy_model,
        patch(
            "vllm.snapshot.distributed.destroy_distributed_environment"
        ) as destroy_world,
        patch("vllm.snapshot.distributed.reset_group_name_registry") as reset,
    ):
        cleanup_dist_env_for_snapshot()

    destroyer = destroy_model.call_args.args[0]
    destroy_world.assert_called_once_with(destroyer)
    reset.assert_called_once_with()
