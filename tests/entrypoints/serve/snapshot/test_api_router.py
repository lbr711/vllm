# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from vllm.entrypoints.serve.snapshot.api_router import router, snapshot_health
from vllm.entrypoints.serve.snapshot.monitor import SnapshotMonitor
from vllm.v1.engine.exceptions import EngineDeadError


def _request() -> Mock:
    request = Mock()
    request.app.state = SimpleNamespace(
        engine_client=SimpleNamespace(check_health=AsyncMock()),
        snapshot_monitor=SnapshotMonitor(),
    )
    return request


def test_snapshot_router_owns_lifecycle_endpoints():
    paths = {route.path for route in router.routes}

    assert paths == {
        "/snapshot/health",
        "/suspend",
        "/resume",
        "/device_unlock",
    }


@pytest.mark.asyncio
async def test_snapshot_health_waits_for_suspend_on_cold_start():
    request = _request()

    with patch(
        "vllm.entrypoints.serve.snapshot.api_router."
        "is_restored_from_host_side_snapshot",
        return_value=False,
    ):
        response = await snapshot_health(request)
        assert response.status_code == HTTPStatus.ACCEPTED

        request.app.state.snapshot_monitor.mark_suspend_done()
        response = await snapshot_health(request)

    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_snapshot_health_waits_for_resume_after_restore():
    request = _request()
    request.app.state.snapshot_monitor.mark_suspend_done()

    with patch(
        "vllm.entrypoints.serve.snapshot.api_router."
        "is_restored_from_host_side_snapshot",
        return_value=True,
    ):
        response = await snapshot_health(request)
        assert response.status_code == HTTPStatus.ACCEPTED

        request.app.state.snapshot_monitor.mark_resume_done()
        response = await snapshot_health(request)

    assert response.status_code == HTTPStatus.OK


@pytest.mark.asyncio
async def test_snapshot_health_reports_dead_engine():
    request = _request()
    request.app.state.engine_client.check_health.side_effect = EngineDeadError()

    response = await snapshot_health(request)

    assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
