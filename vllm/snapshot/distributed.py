# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    reset_group_name_registry,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _abort_hccl_process_group(process_group: ProcessGroup) -> None:
    process_group._get_backend(torch.device("npu")).abort_hccl_comm("reinit")


def cleanup_dist_env_for_snapshot(shutdown_ray: bool = False) -> None:
    """Abort HCCL communicators and clear distributed state for restore."""
    destroy_model_parallel(_abort_hccl_process_group)
    logger.info("Snapshot model-parallel groups destroyed")
    destroy_distributed_environment(_abort_hccl_process_group)
    reset_group_name_registry()
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()
