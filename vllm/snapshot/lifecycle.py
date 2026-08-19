# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.logger import init_logger
from vllm.snapshot.kv_transfer import (
    refresh_scheduler_after_resume,
    refresh_scheduler_handshake_metadata_after_resume,
)
from vllm.snapshot.utils import get_local_ip

logger = init_logger(__name__)

_R = TypeVar("_R")


class SnapshotEngine(Protocol):
    vllm_config: Any
    scheduler: Any
    model_executor: Any
    dp_group: Any
    _transport_lock: Any
    _transport_restored: bool

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]: ...

    def _reconnect_transport(self, data_parallel_master_ip: str) -> None: ...


def suspend_engine(
    engine_core: SnapshotEngine,
    model_save_path: str | None,
) -> None:
    logger.info("[snapshot] [engine] prepare workers for suspend")
    engine_core.collective_rpc("snapshot_prepare_suspend", args=(model_save_path,))

    logger.info("[snapshot] [engine] gc.collect()")
    gc.collect()

    logger.info("[snapshot] [engine] suspend workers")
    engine_core.collective_rpc("snapshot_suspend")


def unlock_engine(engine_core: SnapshotEngine) -> None:
    logger.info("[snapshot] [engine] unlock workers")
    engine_core.collective_rpc("snapshot_unlock")


def resume_engine(
    engine_core: SnapshotEngine,
    data_parallel_master_ip: str,
    model_path: str | None,
) -> None:
    # TCP may already have been restored by the EngineCore metadata watcher;
    # IPC reaches this same path and restores its transport here.
    with engine_core._transport_lock:
        if not engine_core._transport_restored:
            engine_core._reconnect_transport(data_parallel_master_ip)
            engine_core._transport_restored = True

    local_ip = get_local_ip()
    parallel_config = engine_core.vllm_config.parallel_config
    parallel_config.data_parallel_master_ip = data_parallel_master_ip

    logger.info("[snapshot] [engine] prepare workers for resume")
    engine_core.collective_rpc(
        "snapshot_prepare_resume",
        args=(local_ip, data_parallel_master_ip),
    )

    dp_group = getattr(engine_core, "dp_group", None)
    if dp_group is not None:
        logger.info("[snapshot] [engine] rebuild EngineCore DP group")
        stateless_destroy_torch_distributed_process_group(dp_group)
        parallel_config._data_parallel_master_port_list.clear()
        engine_core.dp_group = parallel_config.stateless_init_dp_group()
    else:
        logger.info(
            "[snapshot] [engine] skip EngineCore DP group rebuild "
            "(data_parallel_size==1 or non-DPEngineCoreProc)"
        )

    logger.info("[snapshot] [engine] restore worker model state")
    engine_core.collective_rpc("snapshot_restore_model", args=(model_path,))

    logger.info("[snapshot] [engine] refresh scheduler KV state")
    refresh_scheduler_after_resume(engine_core, local_ip)

    kv_config = engine_core.vllm_config.kv_transfer_config
    new_engine_id = str(kv_config.engine_id) if kv_config is not None else None

    logger.info("[snapshot] [engine] rebuild worker KV transfer state")
    engine_core.collective_rpc(
        "snapshot_rebuild_kv_transfer",
        args=(local_ip, new_engine_id),
    )

    logger.info("[snapshot] [engine] refresh scheduler handshake metadata")
    refresh_scheduler_handshake_metadata_after_resume(engine_core)
