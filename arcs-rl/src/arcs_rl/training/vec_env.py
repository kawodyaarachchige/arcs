"""
Build the same vectorized Gymnasium environment that training uses.

Keeping this in one place means evaluation and learning see identical observation and action wiring.
"""

from __future__ import annotations

from typing import Any

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.vec_env import DummyVecEnv

from arcs_rl.envs import ArcsMicroserviceEnv
from arcs_rl.envs.wrappers import FlattenMultiDiscreteActions


def make_vec_env(
    config: dict[str, Any],
    *,
    max_episode_steps: int | None = None,
) -> DummyVecEnv:
    """
    One parallel env (batch size 1) matching ``action.algorithm`` (DQN vs PPO).

    Training leaves episodes “open” because learning counts raw steps. Evaluation passes
    ``max_episode_steps`` so each rollout eventually stops; otherwise our simulator never sets
    ``terminated`` and :func:`evaluate_policy` would not finish.
    """
    algo = str(config["action"]["algorithm"]).lower()

    def _thunk() -> Any:
        env: Any = ArcsMicroserviceEnv(config, algorithm=algo)
        if algo == "dqn":
            env = FlattenMultiDiscreteActions(env)
        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    return DummyVecEnv([_thunk])
