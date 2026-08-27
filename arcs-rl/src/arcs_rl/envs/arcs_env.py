"""
Gymnasium environment: choose retry/backoff/timeout, observe the 12-D state, get a reward.

"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from arcs_rl.envs.backends import MockBackend, SimulationBackend
from arcs_rl.envs.continuous_actions import decode_continuous_action, make_ppo_box_space
from arcs_rl.envs.discrete_actions import (
    DiscreteActionLayout,
    decode_discrete_action,
    discrete_layout_from_config,
    make_multi_discrete_space,
)
from arcs_rl.observation.aggregator import StateAggregator, aggregator_from_config
from arcs_rl.observation.features import STATE_DIM
from arcs_rl.rewards.default import compute_reward


class ArcsMicroserviceEnv(gym.Env):
    """Small MDP that wraps telemetry aggregation plus a pluggable simulator backend."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: dict[str, Any],
        *,
        algorithm: str | None = None,
        backend: SimulationBackend | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self._policy = config["policy"]
        self._reward_cfg = config["reward"]
        self._determinism_seed = int(config["determinism"]["seed"])

        algo = (algorithm or str(config["action"]["algorithm"])).lower()
        if algo not in ("dqn", "ppo"):
            msg = f"algorithm must be 'dqn' or 'ppo', got {algo!r}"
            raise ValueError(msg)
        self._algorithm = algo

        self._owns_backend = backend is None
        self._backend: SimulationBackend
        if backend is None:
            self._backend = MockBackend(self._determinism_seed)
        else:
            self._backend = backend

        self._aggregator: StateAggregator = aggregator_from_config(config)

        self._discrete_layout: DiscreteActionLayout | None
        if self._algorithm == "dqn":
            dqn_cfg = config["action"]["dqn"]
            self._discrete_layout = discrete_layout_from_config(self._policy, dqn_cfg)
            self.action_space = make_multi_discrete_space(self._discrete_layout)
        else:
            self._discrete_layout = None
            self.action_space = make_ppo_box_space(self._policy)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )

        self._t = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        eff_seed = int(seed) if seed is not None else self._determinism_seed
        if self._owns_backend:
            self._backend = MockBackend(eff_seed)
        else:
            self._backend.reset()

        self._aggregator = aggregator_from_config(self._config)
        self._t = 0.0
        obs = self._build_obs()
        return obs, {}

    def step(
        self,
        action: Any,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._algorithm == "dqn":
            assert self._discrete_layout is not None
            retry, bo, tmo = decode_discrete_action(np.asarray(action), self._discrete_layout)
        else:
            retry, bo, tmo = decode_continuous_action(np.asarray(action), self._policy)

        self._aggregator.set_policy_context(
            retry_count=retry,
            backoff_multiplier=bo,
            timeout_ms=tmo,
        )
        outcome = self._backend.step(
            self._t,
            retry=retry,
            backoff_multiplier=bo,
            timeout_ms=tmo,
        )

        self._aggregator.record_latency_ms(self._t, outcome.latency_ms)
        self._aggregator.record_request(self._t, ok=outcome.success)
        self._aggregator.record_load(
            self._t,
            cpu=outcome.cpu_util,
            memory=outcome.memory_util,
            queue_depth=outcome.queue_depth,
        )
        self._aggregator.record_global_rps(self._t, outcome.global_rps)

        obs = self._build_obs()

        r_max = int(self._policy["retry"]["max"])
        retry_overhead = float(retry) / float(r_max) if r_max > 0 else 0.0

        err_raw = self._aggregator.raw_error_rate(self._t)
        cascade_thr = float(self._reward_cfg["cascade_error_rate_threshold"])
        cascade_active = err_raw is not None and err_raw >= cascade_thr

        reward, breakdown = compute_reward(
            success_rate=1.0 if outcome.success else 0.0,
            avg_latency_ms=outcome.latency_ms,
            retry_overhead=retry_overhead,
            overload_penalty=outcome.overload_penalty,
            cascade_active=cascade_active,
            reward_cfg=self._reward_cfg,
        )

        self._t += 1.0

        terminated = False
        truncated = False
        info: dict[str, Any] = {
            "retry": retry,
            "backoff_multiplier": bo,
            "timeout_ms": tmo,
            "reward_breakdown": breakdown,
        }
        return obs, reward, terminated, truncated, info

    def _build_obs(self) -> np.ndarray:
        vec = self._aggregator.build_vector(self._t)
        return vec.astype(np.float32, copy=False)

    def render(self) -> None:
        return None
