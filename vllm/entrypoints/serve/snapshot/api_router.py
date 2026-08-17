# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import Response

from vllm.engine.protocol import EngineClient
from vllm.v1.engine.exceptions import EngineDeadError

from .monitor import SnapshotMonitor
from .utils import is_restored_from_host_side_snapshot

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def snapshot_monitor(request: Request) -> SnapshotMonitor:
    return request.app.state.snapshot_monitor


@router.get("/snapshot/health", response_class=Response)
async def snapshot_health(raw_request: Request) -> Response:
    try:
        await engine_client(raw_request).check_health()
    except EngineDeadError:
        return Response(status_code=HTTPStatus.SERVICE_UNAVAILABLE)

    monitor = snapshot_monitor(raw_request)
    ready = (
        monitor.is_resume_done
        if is_restored_from_host_side_snapshot()
        else monitor.is_suspend_done
    )
    status_code = HTTPStatus.OK if ready else HTTPStatus.ACCEPTED
    return Response(status_code=status_code)


@router.post("/suspend", response_class=Response)
async def suspend(raw_request: Request) -> Response:
    model_save_path = raw_request.query_params.get("model_save_path")
    if model_save_path is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Missing required parameter: model_save_path",
        )
    await engine_client(raw_request).suspend(model_save_path=model_save_path)
    return Response(status_code=HTTPStatus.OK)


@router.post("/resume", response_class=Response)
async def resume(raw_request: Request) -> Response:
    data_parallel_master_ip = raw_request.query_params.get("data_parallel_master_ip")
    model_path = raw_request.query_params.get("model_path")
    if data_parallel_master_ip is None or model_path is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Missing required parameter: data_parallel_master_ip and model_path",
        )
    await engine_client(raw_request).resume(
        data_parallel_master_ip=data_parallel_master_ip,
        model_path=model_path,
    )
    return Response(status_code=HTTPStatus.OK)


@router.post("/device_unlock", response_class=Response)
async def device_unlock(raw_request: Request) -> Response:
    await engine_client(raw_request).device_unlock()
    return Response(status_code=HTTPStatus.OK)


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
