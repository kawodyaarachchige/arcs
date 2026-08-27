"""
Run the exported neural net on a 12-D observation and read off retry / backoff / timeout.

Training uses either a flat DQN index or a PPO box; this file mirrors that for serving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from arcs_rl.envs.continuous_actions import decode_continuous_action
from arcs_rl.envs.discrete_actions import (
    DiscreteActionLayout,
    decode_discrete_action,
    discrete_layout_from_config,
    make_multi_discrete_space,
)
from arcs_rl.envs.flat_actions import flat_to_multi, nvec_from_space
from arcs_rl.observation.features import STATE_DIM
from arcs_rl.safeguards import PolicyAction


class TorchInferenceRuntime:
    """
    Holds a TorchScript module plus the action layout needed to decode its outputs.

    The YAML file decides whether load this at all; skipped entirely when inference is off.
    """

    def __init__(
        self,
        *,
        root_config: dict[str, Any],
        torchscript_path: Path,
        sidecar: dict[str, Any],
        model: Any,
        device_label: str,
    ) -> None:
        self._policy_cfg = root_config["policy"]
        self._action_cfg = root_config["action"]
        self._algo = str(sidecar["algorithm"]).lower()
        self._model = model
        self._torchscript_path = torchscript_path
        self._device_label = device_label
        self._discrete_layout: DiscreteActionLayout | None = None
        self._nvec: np.ndarray | None = None
        if self._algo == "dqn":
            self._discrete_layout = discrete_layout_from_config(
                self._policy_cfg,
                self._action_cfg["dqn"],
            )
            md = make_multi_discrete_space(self._discrete_layout)
            self._nvec = nvec_from_space(md)
        elif self._algo == "ppo":
            self._discrete_layout = None
            self._nvec = None
        else:
            msg = f"Unsupported algorithm in sidecar: {self._algo!r}"
            raise ValueError(msg)

    def suggested_action(self, observation: np.ndarray) -> PolicyAction:
        """
        Run one forward pass and convert the network output into a concrete policy triple.

        `observation` must be length STATE_DIM (twelve normalized features in training order).
        """
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.size != STATE_DIM:
            msg = f"observation must have length {STATE_DIM}, got {obs.size}"
            raise ValueError(msg)
        import torch

        device = torch.device(self._device_label if self._device_label == "cuda" else "cpu")
        batch = torch.from_numpy(obs).unsqueeze(0).to(device)
        if self._algo == "dqn":
            assert self._discrete_layout is not None and self._nvec is not None
            q = self._model(batch)
            flat = int(torch.argmax(q, dim=-1).item())
            branch = flat_to_multi(self._nvec, flat)
            r, b, t = decode_discrete_action(branch, self._discrete_layout)
            return PolicyAction(retry=r, backoff_multiplier=b, timeout_ms=t)
        assert self._algo == "ppo"
        mean = self._model(batch).detach().cpu().numpy().reshape(-1)
        r, b, t = decode_continuous_action(mean, self._policy_cfg)
        return PolicyAction(retry=r, backoff_multiplier=b, timeout_ms=t)


def load_torch_inference_or_none(root_config: dict[str, Any]) -> TorchInferenceRuntime | None:
    """
    Load TorchScript from paths in `serving`, or return None when inference is disabled.

    Raises if inference is on but the TorchScript file is missing and the config says to fail hard.
    """
    serving = root_config["serving"]
    if not serving["inference_enabled"]:
        return None
    raw = serving["torchscript_path"]
    if not raw:
        if serving["fail_if_model_missing"]:
            msg = "serving.inference_enabled is true but serving.torchscript_path is empty"
            raise ValueError(msg)
        return None
    ts_path = Path(raw)
    if not ts_path.is_file():
        if serving["fail_if_model_missing"]:
            msg = f"TorchScript file not found: {ts_path}"
            raise FileNotFoundError(msg)
        return None
    sidecar_path = ts_path.with_suffix(".json")
    if not sidecar_path.is_file():
        msg = f"Sidecar JSON missing next to model: {sidecar_path}"
        raise FileNotFoundError(msg)
    meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    import torch

    device_label = str(serving["device"])
    device = torch.device(device_label if device_label == "cuda" else "cpu")
    model = torch.jit.load(str(ts_path), map_location=device)
    model.eval()
    return TorchInferenceRuntime(
        root_config=root_config,
        torchscript_path=ts_path,
        sidecar=meta,
        model=model,
        device_label=device_label,
    )
