# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.engine.core_client import AsyncMPClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_reconnect,expected_order",
    [
        (True, ["ready", "resume"]),
        (False, ["resume", "ready"]),
    ],
)
async def test_resume_engine_order_matches_transport(
    transport_reconnect,
    expected_order,
):
    calls = []

    async def wait_for_engines_ready():
        calls.append("ready")

    async def call_utility_async(method, data_parallel_master_ip, model_path):
        assert method == "resume"
        assert data_parallel_master_ip == "10.0.0.2"
        assert model_path == "/snapshot/model"
        calls.append("resume")

    client = SimpleNamespace(
        _snapshot_transport_reconnect=transport_reconnect,
        wait_for_engines_ready=wait_for_engines_ready,
        call_utility_async=call_utility_async,
    )

    await AsyncMPClient._resume_engines(
        client,
        "10.0.0.2",
        "/snapshot/model",
    )

    assert calls == expected_order
