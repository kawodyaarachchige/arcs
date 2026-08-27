"""
Turn a saved Stable-Baselines3 zip into TorchScript for low-latency inference.

DQN exports the Q-network (one scalar Q per flat action). PPO exports the actor head that picks
actions. ONNX is optional so installs without extra packages stay small.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN, PPO


def write_model_sidecar(
    path: Path,
    *,
    algorithm: str,
    observation_dim: int,
    action_summary: dict[str, Any],
) -> None:
    """Write a small JSON next to the artifact so servers know how to interpret outputs."""
    meta = {
        "algorithm": algorithm,
        "observation_dim": observation_dim,
        "action": action_summary,
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def export_torchscript(
    sb3_zip: str | Path,
    out_ts: str | Path,
    *,
    algorithm: str | None = None,
    observation_dim: int | None = None,
) -> Path:
    """
    Load `sb3_zip` and save TorchScript to `out_ts`. Also writes `out_ts.with_suffix('.json')`.

    `algorithm` defaults to inferring from the file name or class stored in the zip.
    """
    sb3_zip = Path(sb3_zip)
    out_ts = Path(out_ts)
    out_ts.parent.mkdir(parents=True, exist_ok=True)

    algo = (algorithm or "").lower()
    if not algo:
        # Guess from common filenames; caller should pass algorithm when ambiguous.
        name = sb3_zip.name.lower()
        if "dqn" in name:
            algo = "dqn"
        elif "ppo" in name:
            algo = "ppo"
        else:
            msg = "Pass algorithm='dqn' or 'ppo' when the zip path does not contain a hint."
            raise ValueError(msg)

    obs_dim: int | None = observation_dim
    device = torch.device("cpu")
    summary: dict[str, Any]

    if algo == "dqn":
        dqn = DQN.load(sb3_zip, device=device)
        osh = dqn.observation_space.shape
        if osh is None:
            msg = "DQN observation_space.shape is missing"
            raise ValueError(msg)
        if obs_dim is None:
            obs_dim = int(osh[0])
        dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
        q_net = dqn.q_net
        q_net.eval()
        traced = torch.jit.trace(q_net, dummy)
        traced.save(str(out_ts))
        act = dqn.action_space
        if not isinstance(act, Discrete):
            msg = "Exported DQN expects a Discrete action space (flattened MultiDiscrete)."
            raise TypeError(msg)
        summary = {
            "kind": "dqn_flat_discrete",
            "num_actions": int(act.n),
            "note": "Output is shape (1, n_actions) Q-values; argmax picks the flat action index.",
        }
    elif algo == "ppo":
        ppo = PPO.load(sb3_zip, device=device)
        osh = ppo.observation_space.shape
        if osh is None:
            msg = "PPO observation_space.shape is missing"
            raise ValueError(msg)
        if obs_dim is None:
            obs_dim = int(osh[0])
        dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
        policy = ppo.policy
        policy.eval()

        class ActorTrace(torch.nn.Module):
            """Wrap policy forward to return action mean only (no stochastic sampling)."""

            def __init__(self, pol: Any) -> None:
                super().__init__()
                self.pol = pol

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                dist = self.pol.get_distribution(obs)
                raw = cast(Any, dist.distribution)
                return raw.mean

        wrapper = ActorTrace(policy)
        wrapper.eval()
        traced = torch.jit.trace(wrapper, dummy)
        traced.save(str(out_ts))
        box = ppo.action_space
        if not isinstance(box, Box):
            msg = "Exported PPO expects a Box action space."
            raise TypeError(msg)
        bsh = box.shape
        if bsh is None:
            msg = "PPO Box shape is missing"
            raise ValueError(msg)
        summary = {
            "kind": "ppo_box_mean",
            "action_shape": list(bsh),
            "note": "Output is the Gaussian mean vector (same shape as the Box action space).",
        }
    else:
        msg = f"Unsupported algorithm for export: {algo!r}"
        raise ValueError(msg)

    assert obs_dim is not None
    write_model_sidecar(
        out_ts.with_suffix(".json"),
        algorithm=algo,
        observation_dim=obs_dim,
        action_summary=summary,
    )
    return out_ts


def export_onnx_if_available(
    sb3_zip: str | Path,
    out_onnx: str | Path,
    *,
    algorithm: str,
    observation_dim: int | None = None,
) -> Path | None:
    """
    Optional ONNX export. Returns None if `onnx` is not installed.

    Install the `onnx` extra from the package metadata to use this in production pipelines.
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        return None

    sb3_zip = Path(sb3_zip)
    out_onnx = Path(out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    if algorithm.lower() == "dqn":
        dqn_m = DQN.load(sb3_zip, device=device)
        osh = dqn_m.observation_space.shape
        if osh is None:
            return None
        obs_dim = observation_dim or int(osh[0])
        dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
        torch.onnx.export(
            dqn_m.q_net,
            (dummy,),
            str(out_onnx),
            input_names=["obs"],
            output_names=["q_values"],
            dynamic_axes={"obs": {0: "batch"}, "q_values": {0: "batch"}},
        )
        return out_onnx
    if algorithm.lower() == "ppo":
        ppo_m = PPO.load(sb3_zip, device=device)
        osh = ppo_m.observation_space.shape
        if osh is None:
            return None
        obs_dim = observation_dim or int(osh[0])
        dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
        pol = ppo_m.policy

        class ActorMean(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pol = pol

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                dist = self.pol.get_distribution(obs)
                raw = cast(Any, dist.distribution)
                return raw.mean

        am = ActorMean()
        am.eval()
        torch.onnx.export(
            am,
            (dummy,),
            str(out_onnx),
            input_names=["obs"],
            output_names=["action_mean"],
            dynamic_axes={"obs": {0: "batch"}, "action_mean": {0: "batch"}},
        )
        return out_onnx
    return None
