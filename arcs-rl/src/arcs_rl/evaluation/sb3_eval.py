"""
Run a trained Stable-Baselines3 agent on the mock microservice environment.

This measures simulator reward only. It does not benchmark the policy gRPC service or Envoy latency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

from arcs_rl.config import effective_training_seed
from arcs_rl.training._device import resolve_training_device
from arcs_rl.training.vec_env import make_vec_env

# Long enough for a meaningful return; short enough that eval stays snappy on a laptop.
_DEFAULT_EVAL_HORIZON = 200


def evaluate_saved_model(
    config: dict[str, Any],
    model_zip: str | Path,
    *,
    n_eval_episodes: int = 5,
    deterministic: bool = True,
    max_episode_steps: int = _DEFAULT_EVAL_HORIZON,
) -> tuple[float, float]:
    """
    Load a ``.zip`` checkpoint and estimate mean / std episode return.

    Uses the same vectorized env as training so results stay comparable to learning curves.
    """
    model_path = Path(model_zip)
    if not model_path.is_file():
        msg = f"Model file not found: {model_path}"
        raise FileNotFoundError(msg)

    seed = effective_training_seed(config)
    tcfg = config["training"]
    device = resolve_training_device(str(tcfg["device"]))
    set_random_seed(seed, using_cuda=device == "cuda")

    vec = make_vec_env(config, max_episode_steps=max_episode_steps)
    algo = str(config["action"]["algorithm"]).lower()

    try:
        if algo == "dqn":
            model = DQN.load(str(model_path), env=vec, device=device)
        elif algo == "ppo":
            model = PPO.load(str(model_path), env=vec, device=device)
        else:
            msg = f"Unsupported action.algorithm for evaluation: {algo!r}"
            raise ValueError(msg)

        mean_reward, std_reward = evaluate_policy(
            model,
            vec,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            warn=False,
        )
    finally:
        vec.close()

    return float(mean_reward), float(std_reward)
