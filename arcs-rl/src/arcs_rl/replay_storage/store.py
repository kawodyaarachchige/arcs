"""
Save transitions as numpy .npz shards plus a small JSON index.

DQN stores each action as three integers (one per MultiDiscrete branch). PPO stores three floats
(the Box sample). This matches what Stable-Baselines3 puts in the replay buffer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete

INDEX_FILENAME = "index.json"
SCHEMA_VERSION = 1


@dataclass
class _ShardInfo:
    name: str
    n: int


class ReplayStorage:
    """
    Append transitions, flush them into fixed-size .npz files, and drop oldest files when the
    shard count would exceed `max_shards` (rolling window on disk).
    """

    def __init__(
        self,
        root: Path,
        *,
        transitions_per_shard: int,
        max_shards: int,
        algorithm: str,
        state_dim: int,
        action_shape: tuple[int, ...],
        action_numpy_kind: str,
        metrics_exporter: Any | None = None,
    ) -> None:
        if transitions_per_shard < 1 or max_shards < 1:
            msg = "transitions_per_shard and max_shards must be >= 1"
            raise ValueError(msg)
        self._root = root
        self._tps = transitions_per_shard
        self._max_shards = max_shards
        self._algorithm = algorithm
        self._state_dim = state_dim
        self._action_shape = action_shape
        self._action_kind = action_numpy_kind
        self._root.mkdir(parents=True, exist_ok=True)

        self._index_path = self._root / INDEX_FILENAME
        self._shards: list[_ShardInfo] = []
        self._pending: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]] = []
        self._next_shard_id = 0
        # Optional hook: push replay sizes to Prometheus when training metrics are enabled.
        self._metrics = metrics_exporter
        self._load_or_init_index()
        self._sync_replay_metrics()

    def _load_or_init_index(self) -> None:
        if not self._index_path.is_file():
            self._save_index()
            return
        raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != SCHEMA_VERSION:
            msg = f"Unsupported replay index schema_version: {raw.get('schema_version')}"
            raise ValueError(msg)
        if raw["algorithm"] != self._algorithm:
            msg = (
                f"Existing replay index algorithm is {raw['algorithm']!r} but "
                f"this open uses {self._algorithm!r}; use a fresh directory or match the algorithm."
            )
            raise ValueError(msg)
        if tuple(raw["obs_shape"]) != (self._state_dim,):
            msg = "Existing replay index obs_shape does not match this environment."
            raise ValueError(msg)
        if tuple(raw["action_shape"]) != self._action_shape:
            msg = "Existing replay index action_shape does not match this environment."
            raise ValueError(msg)
        self._next_shard_id = int(raw.get("next_shard_id", 0))
        for s in raw["shards"]:
            self._shards.append(_ShardInfo(name=str(s["name"]), n=int(s["n"])))

    def _sync_replay_metrics(self) -> None:
        """Tell the metrics helper how big the buffer is (no-op if metrics are off)."""
        if self._metrics is not None:
            self._metrics.sync_replay_from_storage(self)

    @property
    def total_transitions(self) -> int:
        """How many transitions are stored on disk (not counting the in-memory pending batch)."""
        return sum(s.n for s in self._shards)

    @property
    def pending_count(self) -> int:
        """Transitions buffered in RAM waiting for the next shard flush."""
        return len(self._pending)

    def append(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Record one transition. Shapes must match this storage's obs and action layout."""
        obs = np.asarray(obs, dtype=np.float32).reshape(self._state_dim)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(self._state_dim)
        action = np.asarray(action)
        expected = self._action_shape
        if action.shape != expected:
            msg = f"action shape {action.shape} != expected {expected}"
            raise ValueError(msg)
        if self._action_kind == "int":
            action = action.astype(np.int32, copy=False)
        else:
            action = action.astype(np.float32, copy=False)
        self._pending.append((obs, next_obs, action, float(reward), bool(done)))
        self._sync_replay_metrics()
        if len(self._pending) >= self._tps:
            self._flush_pending()

    def flush(self) -> None:
        """Write any pending transitions to disk (partial shard allowed)."""
        if self._pending:
            self._flush_pending(force=True)

    def _rotate_if_needed(self) -> None:
        # While at capacity and need room for another shard file, delete the oldest.
        while len(self._shards) >= self._max_shards:
            oldest = self._shards.pop(0)
            path = self._root / oldest.name
            if path.is_file():
                path.unlink()
            if self._metrics is not None:
                self._metrics.on_replay_shard_removed()
            self._save_index()

    def _flush_pending(self, *, force: bool = False) -> None:
        while self._pending:
            if not force and len(self._pending) < self._tps:
                break
            n = len(self._pending) if force else self._tps
            chunk = self._pending[:n]
            self._pending = self._pending[n:]

            obs_a = np.stack([c[0] for c in chunk], axis=0)
            next_a = np.stack([c[1] for c in chunk], axis=0)
            act_a = np.stack([c[2] for c in chunk], axis=0)
            rew_a = np.array([c[3] for c in chunk], dtype=np.float32)
            done_a = np.array([c[4] for c in chunk], dtype=np.bool_)

            self._rotate_if_needed()
            name = f"shard_{self._next_shard_id:05d}.npz"
            self._next_shard_id += 1
            path = self._root / name
            np.savez_compressed(
                path,
                obs=obs_a,
                next_obs=next_a,
                actions=act_a,
                rewards=rew_a,
                dones=done_a,
            )
            self._shards.append(_ShardInfo(name=name, n=len(chunk)))
            if self._metrics is not None:
                self._metrics.on_replay_flush()
            self._save_index()
            self._sync_replay_metrics()
            if force:
                break

    def _save_index(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "algorithm": self._algorithm,
            "obs_shape": [self._state_dim],
            "action_shape": list(self._action_shape),
            "action_storage": self._action_kind,
            "shards": [{"name": s.name, "n": s.n} for s in self._shards],
            "next_shard_id": self._next_shard_id,
            "total_transitions": self.total_transitions,
        }
        self._index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_arrays(self) -> dict[str, np.ndarray]:
        """
        Load every transition from disk into big numpy arrays (may use a lot of RAM).

        Pending (not yet flushed) rows are included so training always sees the latest data.
        """
        obs_parts: list[np.ndarray] = []
        next_parts: list[np.ndarray] = []
        act_parts: list[np.ndarray] = []
        rew_parts: list[np.ndarray] = []
        done_parts: list[np.ndarray] = []

        for s in self._shards:
            path = self._root / s.name
            data = np.load(path)
            obs_parts.append(data["obs"])
            next_parts.append(data["next_obs"])
            act_parts.append(data["actions"])
            rew_parts.append(data["rewards"])
            done_parts.append(data["dones"])

        if self._pending:
            obs_parts.append(np.stack([p[0] for p in self._pending], axis=0))
            next_parts.append(np.stack([p[1] for p in self._pending], axis=0))
            act_parts.append(np.stack([p[2] for p in self._pending], axis=0))
            rew_parts.append(np.array([p[3] for p in self._pending], dtype=np.float32))
            done_parts.append(np.array([p[4] for p in self._pending], dtype=np.bool_))

        if not obs_parts:
            adt = np.int32 if self._action_kind == "int" else np.float32
            return {
                "obs": np.array([], dtype=np.float32).reshape(0, self._state_dim),
                "next_obs": np.array([], dtype=np.float32).reshape(0, self._state_dim),
                "actions": np.array([], dtype=adt).reshape(0, *self._action_shape),
                "rewards": np.array([], dtype=np.float32),
                "dones": np.array([], dtype=np.bool_),
            }

        return {
            "obs": np.concatenate(obs_parts, axis=0),
            "next_obs": np.concatenate(next_parts, axis=0),
            "actions": np.concatenate(act_parts, axis=0),
            "rewards": np.concatenate(rew_parts, axis=0),
            "dones": np.concatenate(done_parts, axis=0),
        }


def open_storage(
    root: str | Path,
    config: dict[str, Any],
    *,
    env: Env,
    metrics_exporter: Any | None = None,
) -> ReplayStorage:
    """
    Open or create storage under `replay_buffer.path`, using spaces from `env` for shapes.

    Callers should use the same Gymnasium environment class as training so dimensions line up.
    """
    rb = config["replay_buffer"]
    root_p = Path(rb["path"]).expanduser()
    algo = str(config["action"]["algorithm"]).lower()
    if algo not in ("dqn", "ppo"):
        msg = f"replay storage needs action.algorithm dqn or ppo, got {algo!r}"
        raise ValueError(msg)

    obs_shp = env.observation_space.shape
    if obs_shp is None:
        msg = "observation_space.shape is required for replay storage"
        raise ValueError(msg)
    obs_dim = int(obs_shp[0])
    act_space = env.action_space
    action_numpy_kind = "float"
    action_shape: tuple[int, ...]
    if algo == "dqn":
        if not isinstance(act_space, MultiDiscrete):
            msg = "DQN replay storage expects a MultiDiscrete action space"
            raise TypeError(msg)
        # We store one flat index per row so files match Stable-Baselines3 DQN (Discrete only).
        action_shape = (1,)
        action_numpy_kind = "int"
    else:
        if not isinstance(act_space, Box):
            msg = "PPO replay storage expects a Box action space"
            raise TypeError(msg)
        ash = act_space.shape
        if ash is None:
            msg = "Box action_space.shape is required"
            raise ValueError(msg)
        action_shape = tuple(int(x) for x in ash)

    return ReplayStorage(
        root_p,
        transitions_per_shard=int(rb["transitions_per_shard"]),
        max_shards=int(rb["max_shards"]),
        algorithm=algo,
        state_dim=obs_dim,
        action_shape=action_shape,
        action_numpy_kind=action_numpy_kind,
        metrics_exporter=metrics_exporter,
    )
