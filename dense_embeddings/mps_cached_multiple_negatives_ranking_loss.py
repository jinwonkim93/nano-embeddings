from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss


class _MPSRandContext:
    """
    RNG context manager that works across CUDA/MPS without relying on checkpoint
    helpers that assume a `torch.<device>.device(...)` context manager exists.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.device_type, self.device_ids = self._collect_device_ids(*tensors)
        self.fwd_device_states = self._get_device_states(self.device_type, self.device_ids)
        self._fork = None

    @staticmethod
    def _collect_device_ids(*args) -> tuple[str | None, list[int]]:
        device_type = None
        device_ids: list[int] = []

        def visit(obj: Any) -> None:
            nonlocal device_type, device_ids
            if isinstance(obj, dict):
                for value in obj.values():
                    visit(value)
                return
            if isinstance(obj, (list, tuple)):
                for value in obj:
                    visit(value)
                return
            if not isinstance(obj, torch.Tensor):
                return
            if obj.device.type in {"cpu", "meta"}:
                return

            if device_type is None:
                device_type = obj.device.type
            elif obj.device.type != device_type:
                return

            device_id = obj.device.index if obj.device.index is not None else 0
            if device_id not in device_ids:
                device_ids.append(device_id)

        visit(args)
        return device_type, device_ids

    @staticmethod
    def _get_device_states(device_type: str | None, device_ids: list[int]) -> list[Tensor]:
        if device_type is None or not device_ids:
            return []

        device_module = getattr(torch, device_type, None)
        if device_module is None or not hasattr(device_module, "get_rng_state"):
            return []

        states: list[Tensor] = []
        for device_id in device_ids:
            try:
                states.append(device_module.get_rng_state(device_id))
            except TypeError:
                states.append(device_module.get_rng_state())
        return states

    @staticmethod
    def _set_device_states(device_type: str | None, device_ids: list[int], states: list[Tensor]) -> None:
        if device_type is None or not device_ids or not states:
            return

        device_module = getattr(torch, device_type, None)
        if device_module is None or not hasattr(device_module, "set_rng_state"):
            return

        for device_id, state in zip(device_ids, states, strict=False):
            try:
                device_module.set_rng_state(state, device_id)
            except TypeError:
                device_module.set_rng_state(state)

    def __enter__(self) -> None:
        if self.device_type is not None:
            self._fork = torch.random.fork_rng(
                devices=self.device_ids,
                enabled=True,
                device_type=self.device_type,
            )
            self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        self._set_device_states(self.device_type, self.device_ids, self.fwd_device_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._fork is not None:
            self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


class MPSCachedMultipleNegativesRankingLoss(CachedMultipleNegativesRankingLoss):
    """
    Drop-in MPS-safe variant of CachedMultipleNegativesRankingLoss.

    This only overrides minibatch embedding RNG-state handling.
    """

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: _MPSRandContext | None = None,
    ) -> tuple[Tensor, _MPSRandContext | None]:
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            key: value[begin:end] if isinstance(value, torch.Tensor) else value
            for key, value in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                copied_state = _MPSRandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]
        return reps, copied_state
