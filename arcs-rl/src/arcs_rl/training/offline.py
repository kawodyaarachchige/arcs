"""
Train from data on disk (or fill synthetic data first) using gradient updates.

Stable-Baselines3's DQN only supports Discrete actions, so use a wrapper that flattens
MultiDiscrete knobs into one index. PPO is on-policy in SB3, so offline gradient training is only
implemented for DQN here; use online training for PPO.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from arcs_rl.config import effective_training_seed
from arcs_rl.envs import ArcsMicroserviceEnv
from arcs_rl.envs.wrappers import FlattenMultiDiscreteActions
from arcs_rl.monitoring import maybe_start_training_metrics
from arcs_rl.replay_storage import ReplayStorage, open_storage
from arcs_rl.training._device import resolve_training_device
from arcs_rl.training.vec_env import make_vec_env


def _fill_storage(
    config: dict[str, Any],
    storage: ReplayStorage,
    train_env: Any,
    *,
    seed: int,
) -> int:
    """
    Random play until reach min_transitions or hit fill_rollout_max_steps.

    Synthetic data only: the mock backend is deterministic given the seed. Importing real traces
    should stay a separate, explicit step so teams remember privacy and retention rules.
    """
    rb = config["replay_buffer"]
    off = config["training"]["offline"]
    min_rows = int(rb["min_transitions"])
    cap = int(off["fill_rollout_max_steps"])

    rng = np.random.default_rng(seed)
    obs, _ = train_env.reset(seed=seed)
    steps = 0
    while storage.total_transitions + storage.pending_count < min_rows and steps < cap:
        a = int(rng.integers(0, train_env.action_space.n))
        next_obs, reward, terminated, truncated, _ = train_env.step(a)
        done = bool(terminated or truncated)
        act_arr = np.array([a], dtype=np.int32)
        storage.append(obs, next_obs, act_arr, float(reward), done)
        steps += 1
        if done:
            obs, _ = train_env.reset()
        else:
            obs = next_obs
    storage.flush()
    return steps


def _populate_sb3_replay_dqn(model: DQN, arrays: dict[str, np.ndarray]) -> None:
    """Copy numpy shards into the algorithm's replay memory."""
    d = arrays
    n = d["obs"].shape[0]
    if n == 0:
        msg = "No transitions to learn from; widen fill_rollout_max_steps or lower min_transitions."
        raise ValueError(msg)
    buf = model.replay_buffer
    if buf is None:
        msg = "DQN replay buffer was not initialized"
        raise RuntimeError(msg)
    buf.reset()
    for i in range(n):
        buf.add(
            d["obs"][i : i + 1],
            d["next_obs"][i : i + 1],
            d["actions"][i : i + 1].reshape(1, 1),
            np.array([d["rewards"][i]], dtype=np.float32),
            np.array([d["dones"][i]], dtype=np.float32),
            [{}],
        )


def run_offline_training(
    config: dict[str, Any],
    *,
    run_name: str = "offline_run",
) -> Path:
    """
    Fill the on-disk buffer if needed, then run DQN gradient steps on that data.

    Returns the path to the saved SB3 zip file.
    """
    algo = str(config["action"]["algorithm"]).lower()
    if algo != "dqn":
        msg = (
            "Offline gradient training is implemented for DQN only. "
            "PPO in Stable-Baselines3 is on-policy; use online training for PPO."
        )
        raise ValueError(msg)

    seed = effective_training_seed(config)
    try:
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as e:
        msg = "stable_baselines3 is required for training"
        raise ImportError(msg) from e

    use_cuda = resolve_training_device(config["training"]["device"]) == "cuda"
    set_random_seed(seed, using_cuda=use_cuda)

    metrics = maybe_start_training_metrics(config)
    inner = ArcsMicroserviceEnv(config, algorithm="dqn")
    storage = open_storage(
        Path(config["replay_buffer"]["path"]).expanduser(),
        config,
        env=inner,
        metrics_exporter=metrics,
    )
    wenv = FlattenMultiDiscreteActions(inner)

    min_rows = int(config["replay_buffer"]["min_transitions"])
    if storage.total_transitions + storage.pending_count < min_rows:
        _fill_storage(config, storage, wenv, seed=seed)

    arrays = storage.load_arrays()
    n = int(arrays["obs"].shape[0])
    if n < min_rows:
        msg = (
            f"Only {n} transitions after filling; need at least {min_rows}. "
            "Increase training.offline.fill_rollout_max_steps or lower "
            "replay_buffer.min_transitions."
        )
        raise ValueError(msg)

    tcfg = config["training"]
    off = tcfg["offline"]
    dqn_cfg = tcfg["dqn"]
    buf_sz = max(int(dqn_cfg["buffer_size"]), n)
    device = resolve_training_device(str(tcfg["device"]))

    vec = make_vec_env(config)
    model = DQN(
        "MlpPolicy",
        vec,
        learning_rate=float(tcfg["learning_rate"]),
        buffer_size=buf_sz,
        learning_starts=0,
        train_freq=1,
        exploration_fraction=float(dqn_cfg["exploration_fraction"]),
        exploration_initial_eps=float(dqn_cfg["exploration_initial_eps"]),
        exploration_final_eps=float(dqn_cfg["exploration_final_eps"]),
        verbose=0,
        seed=seed,
        device=device,
    )
    # SB3 only wires the training logger during learn(); offline training calls train() directly.
    model.set_logger(configure(tempfile.mkdtemp(), ["stdout"]))
    model._setup_model()
    _populate_sb3_replay_dqn(model, arrays)
    model.train(
        gradient_steps=int(off["gradient_steps"]),
        batch_size=int(off["batch_size"]),
    )

    out_dir = Path(tcfg["checkpoint_dir"]).expanduser() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dqn_offline.zip"
    model.save(str(out_path))
    return out_path
