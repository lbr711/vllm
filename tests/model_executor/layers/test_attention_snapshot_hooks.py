from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention


@pytest.mark.parametrize("attention_cls", [Attention, MLAAttention])
def test_snapshot_hooks_are_forwarded_to_attention_impl(attention_cls):
    impl = MagicMock()
    layer = SimpleNamespace(impl=impl)

    attention_cls.restore_snapshot_derived_state(layer, torch.bfloat16)
    attention_cls.reset_snapshot_runtime_state(layer)

    impl.restore_snapshot_derived_state.assert_called_once_with(torch.bfloat16)
    impl.reset_snapshot_runtime_state.assert_called_once_with()


@pytest.mark.parametrize("attention_cls", [Attention, MLAAttention])
def test_snapshot_hooks_allow_impl_without_snapshot_state(attention_cls):
    layer = SimpleNamespace(impl=object())

    attention_cls.restore_snapshot_derived_state(layer, torch.bfloat16)
    attention_cls.reset_snapshot_runtime_state(layer)
