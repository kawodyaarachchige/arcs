"""Read the project YAML settings file and check it has every section we expect."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from arcs_rl.observation.features import FEATURE_NAMES

REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "version",
        "determinism",
        "policy",
        "action",
        "reward",
        "safeguards",
        "prometheus",
        "replay_buffer",
        "training",
        "simulation",
        "observation",
        "serving",
    }
)

REQUIRED_SERVING_KEYS = frozenset(
    {
        "grpc_port",
        "inference_enabled",
        "torchscript_path",
        "fail_if_model_missing",
        "device",
    }
)
VALID_SERVING_DEVICES = frozenset({"cpu", "cuda"})

VALID_REPLAY_SHARD_FORMATS = frozenset({"numpy"})
VALID_TRAINING_DEVICES = frozenset({"auto", "cpu", "cuda"})

REQUIRED_REPLAY_BUFFER_KEYS = frozenset(
    {
        "path",
        "min_transitions",
        "shard_format",
        "transitions_per_shard",
        "max_shards",
    }
)

REQUIRED_TRAINING_KEYS = frozenset(
    {
        "device",
        "checkpoint_dir",
        "seed",
        "learning_rate",
        "dqn",
        "ppo",
        "online",
        "offline",
    }
)

REQUIRED_TRAINING_DQN_KEYS = frozenset(
    {
        "exploration_fraction",
        "exploration_initial_eps",
        "exploration_final_eps",
        "buffer_size",
    }
)

REQUIRED_TRAINING_PPO_KEYS = frozenset({"n_steps", "batch_size"})

REQUIRED_TRAINING_ONLINE_KEYS = frozenset({"total_timesteps"})

REQUIRED_TRAINING_OFFLINE_KEYS = frozenset(
    {"gradient_steps", "batch_size", "fill_rollout_max_steps"}
)

REQUIRED_SAFEGUARDS_KEYS = frozenset(
    {
        "max_retries",
        "backoff_multiplier_bounds",
        "timeout_ms_bounds",
        "max_policy_changes_per_route_per_minute",
        "circuit_breaker_error_rate_threshold",
        "circuit_breaker_clear_error_rate_threshold",
        "route_state_ttl_seconds",
    }
)

REQUIRED_REWARD_KEYS = frozenset(
    {
        "weights",
        "cascade_penalty_scale",
        "cascade_error_rate_threshold",
    }
)

REQUIRED_REWARD_WEIGHT_KEYS = frozenset(
    {"success_rate", "latency", "retry_overhead", "overload_penalty"}
)

REQUIRED_ACTION_KEYS = frozenset({"algorithm", "dqn", "ppo"})

VALID_ACTION_ALGORITHMS = frozenset({"dqn", "ppo"})

REQUIRED_OBSERVATION_KEYS = frozenset(
    {
        "latency_window_s",
        "error_window_s",
        "reference_rps",
        "missing_value_strategy",
        "feature_order",
        "normalization",
    }
)

VALID_MISSING_STRATEGIES = frozenset({"neutral", "pessimistic"})
VALID_NORMALIZATION_MODES = frozenset({"min_max", "z_score"})

REQUIRED_PROMETHEUS_KEYS = frozenset(
    {
        "enabled",
        "bind_address",
        "scrape_port",
        "metrics_path",
        "metric_prefix",
    }
)


def effective_training_seed(data: dict[str, Any]) -> int:
    """Return the RNG seed for SB3 and the env: training.seed if set, else determinism.seed."""
    ts = data["training"]["seed"]
    if ts is not None:
        return int(ts)
    return int(data["determinism"]["seed"])


def load_config(path: str | Path) -> dict[str, Any]:
    """Parse a YAML config file and return the root mapping."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"Config root must be a mapping, got {type(data).__name__}"
        raise ValueError(msg)
    return data


def validate_config_keys(data: dict[str, Any]) -> None:
    """Raise if any main config section (policy, reward, etc.) is missing."""
    missing = REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing:
        msg = f"Missing required config keys: {sorted(missing)}"
        raise ValueError(msg)
    validate_observation_config(data["observation"])
    validate_reward_config(data["reward"])
    validate_action_config(data["action"], data["policy"])
    validate_safeguards_config(data["safeguards"], data["policy"])
    validate_replay_buffer_config(data["replay_buffer"])
    validate_training_config(data["training"], data["determinism"])
    validate_prometheus_config(data["prometheus"])
    validate_serving_config(data["serving"])


def validate_prometheus_config(pc: dict[str, Any]) -> None:
    """
    Check training/replay metrics export settings.

    The policy service uses its own env vars, not this YAML block.
    """
    if not isinstance(pc, dict):
        msg = f"prometheus must be a mapping, got {type(pc).__name__}"
        raise ValueError(msg)
    missing = REQUIRED_PROMETHEUS_KEYS - pc.keys()
    if missing:
        msg = f"prometheus missing keys: {sorted(missing)}"
        raise ValueError(msg)
    if not isinstance(pc["enabled"], bool):
        msg = "prometheus.enabled must be true or false"
        raise ValueError(msg)
    if not isinstance(pc["bind_address"], str) or not pc["bind_address"].strip():
        msg = "prometheus.bind_address must be a non-empty string"
        raise ValueError(msg)
    port = int(pc["scrape_port"])
    if not (1 <= port <= 65535):
        msg = "prometheus.scrape_port must be between 1 and 65535"
        raise ValueError(msg)
    mp = pc["metrics_path"]
    if not isinstance(mp, str) or not mp.startswith("/"):
        msg = "prometheus.metrics_path must be a string starting with /"
        raise ValueError(msg)
    pfx = pc["metric_prefix"]
    if not isinstance(pfx, str) or not pfx.strip():
        msg = "prometheus.metric_prefix must be a non-empty string"
        raise ValueError(msg)


def validate_serving_config(sv: dict[str, Any]) -> None:
    """Check gRPC and optional TorchScript inference settings."""
    if not isinstance(sv, dict):
        msg = f"serving must be a mapping, got {type(sv).__name__}"
        raise ValueError(msg)
    missing = REQUIRED_SERVING_KEYS - sv.keys()
    if missing:
        msg = f"serving missing keys: {sorted(missing)}"
        raise ValueError(msg)
    gp = int(sv["grpc_port"])
    if not (1 <= gp <= 65535):
        msg = "serving.grpc_port must be between 1 and 65535"
        raise ValueError(msg)
    dev = sv["device"]
    if not isinstance(dev, str) or dev not in VALID_SERVING_DEVICES:
        msg = f"serving.device must be one of {sorted(VALID_SERVING_DEVICES)}, got {dev!r}"
        raise ValueError(msg)
    ts = sv["torchscript_path"]
    if ts is not None and not isinstance(ts, str):
        msg = "serving.torchscript_path must be null or a string path"
        raise ValueError(msg)


def validate_replay_buffer_config(rb: dict[str, Any]) -> None:
    """Raise if replay disk settings are missing or inconsistent."""
    if not isinstance(rb, dict):
        msg = f"replay_buffer must be a mapping, got {type(rb).__name__}"
        raise ValueError(msg)
    missing = REQUIRED_REPLAY_BUFFER_KEYS - rb.keys()
    if missing:
        msg = f"replay_buffer missing keys: {sorted(missing)}"
        raise ValueError(msg)
    fmt = rb["shard_format"]
    if fmt not in VALID_REPLAY_SHARD_FORMATS:
        allowed = sorted(VALID_REPLAY_SHARD_FORMATS)
        msg = f"replay_buffer.shard_format must be one of {allowed}, got {fmt!r}"
        raise ValueError(msg)
    mt = int(rb["min_transitions"])
    if mt < 1:
        msg = "replay_buffer.min_transitions must be >= 1"
        raise ValueError(msg)
    tps = int(rb["transitions_per_shard"])
    if tps < 1:
        msg = "replay_buffer.transitions_per_shard must be >= 1"
        raise ValueError(msg)
    ms = int(rb["max_shards"])
    if ms < 1:
        msg = "replay_buffer.max_shards must be >= 1"
        raise ValueError(msg)


def validate_training_config(tr: dict[str, Any], determinism: dict[str, Any]) -> None:
    """Raise if training hyperparameters are missing or out of range."""
    if not isinstance(tr, dict):
        msg = f"training must be a mapping, got {type(tr).__name__}"
        raise ValueError(msg)
    missing = REQUIRED_TRAINING_KEYS - tr.keys()
    if missing:
        msg = f"training missing keys: {sorted(missing)}"
        raise ValueError(msg)

    dev = tr["device"]
    if not isinstance(dev, str) or dev not in VALID_TRAINING_DEVICES:
        msg = f"training.device must be one of {sorted(VALID_TRAINING_DEVICES)}, got {dev!r}"
        raise ValueError(msg)

    lr = float(tr["learning_rate"])
    if lr <= 0:
        msg = "training.learning_rate must be > 0"
        raise ValueError(msg)

    seed = tr["seed"]
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        msg = "training.seed must be null or a non-negative integer"
        raise ValueError(msg)

    dqn = tr["dqn"]
    if not isinstance(dqn, dict):
        msg = f"training.dqn must be a mapping, got {type(dqn).__name__}"
        raise ValueError(msg)
    dm = REQUIRED_TRAINING_DQN_KEYS - dqn.keys()
    if dm:
        msg = f"training.dqn missing keys: {sorted(dm)}"
        raise ValueError(msg)
    ef = float(dqn["exploration_fraction"])
    if not (0.0 <= ef <= 1.0):
        msg = "training.dqn.exploration_fraction must be between 0 and 1"
        raise ValueError(msg)
    for name in ("exploration_initial_eps", "exploration_final_eps"):
        v = float(dqn[name])
        if not (0.0 <= v <= 1.0):
            msg = f"training.dqn.{name} must be between 0 and 1"
            raise ValueError(msg)
    bs = int(dqn["buffer_size"])
    if bs < 1:
        msg = "training.dqn.buffer_size must be >= 1"
        raise ValueError(msg)

    ppo = tr["ppo"]
    if not isinstance(ppo, dict):
        msg = f"training.ppo must be a mapping, got {type(ppo).__name__}"
        raise ValueError(msg)
    pm = REQUIRED_TRAINING_PPO_KEYS - ppo.keys()
    if pm:
        msg = f"training.ppo missing keys: {sorted(pm)}"
        raise ValueError(msg)
    ns = int(ppo["n_steps"])
    if ns < 1:
        msg = "training.ppo.n_steps must be >= 1"
        raise ValueError(msg)
    pbs = int(ppo["batch_size"])
    if pbs < 1:
        msg = "training.ppo.batch_size must be >= 1"
        raise ValueError(msg)

    on = tr["online"]
    if not isinstance(on, dict):
        msg = f"training.online must be a mapping, got {type(on).__name__}"
        raise ValueError(msg)
    om = REQUIRED_TRAINING_ONLINE_KEYS - on.keys()
    if om:
        msg = f"training.online missing keys: {sorted(om)}"
        raise ValueError(msg)
    tt = int(on["total_timesteps"])
    if tt < 1:
        msg = "training.online.total_timesteps must be >= 1"
        raise ValueError(msg)

    off = tr["offline"]
    if not isinstance(off, dict):
        msg = f"training.offline must be a mapping, got {type(off).__name__}"
        raise ValueError(msg)
    ofm = REQUIRED_TRAINING_OFFLINE_KEYS - off.keys()
    if ofm:
        msg = f"training.offline missing keys: {sorted(ofm)}"
        raise ValueError(msg)
    gs = int(off["gradient_steps"])
    if gs < 1:
        msg = "training.offline.gradient_steps must be >= 1"
        raise ValueError(msg)
    obs = int(off["batch_size"])
    if obs < 1:
        msg = "training.offline.batch_size must be >= 1"
        raise ValueError(msg)
    fr = int(off["fill_rollout_max_steps"])
    if fr < 1:
        msg = "training.offline.fill_rollout_max_steps must be >= 1"
        raise ValueError(msg)

    if not isinstance(determinism, dict) or "seed" not in determinism:
        msg = "determinism.seed is required when validating training"
        raise ValueError(msg)


def validate_observation_config(obs: dict[str, Any]) -> None:
    """Raise if the observation block is incomplete or inconsistent."""
    if not isinstance(obs, dict):
        msg = f"observation must be a mapping, got {type(obs).__name__}"
        raise ValueError(msg)
    missing = REQUIRED_OBSERVATION_KEYS - obs.keys()
    if missing:
        msg = f"observation missing keys: {sorted(missing)}"
        raise ValueError(msg)
    strat = obs["missing_value_strategy"]
    if strat not in VALID_MISSING_STRATEGIES:
        allowed = sorted(VALID_MISSING_STRATEGIES)
        msg = f"missing_value_strategy must be one of {allowed}, got {strat!r}"
        raise ValueError(msg)
    norm = obs["normalization"]
    if not isinstance(norm, dict):
        msg = f"observation.normalization must be a mapping, got {type(norm).__name__}"
        raise ValueError(msg)
    mode = norm.get("mode")
    if mode not in VALID_NORMALIZATION_MODES:
        msg = f"normalization.mode must be one of {sorted(VALID_NORMALIZATION_MODES)}, got {mode!r}"
        raise ValueError(msg)
    fo = obs["feature_order"]
    if not isinstance(fo, list) or len(fo) != 12:
        msg = "feature_order must be a list of exactly 12 feature names"
        raise ValueError(msg)
    if tuple(fo) != FEATURE_NAMES:
        msg = "feature_order must match FEATURE_NAMES in arcs_rl.observation.features"
        raise ValueError(msg)


def validate_reward_config(reward: dict[str, Any]) -> None:
    """Raise if the reward block cannot drive the scoring function."""
    if not isinstance(reward, dict):
        msg = f"reward must be a mapping, got {type(reward).__name__}"
        raise ValueError(msg)
    missing = REQUIRED_REWARD_KEYS - reward.keys()
    if missing:
        msg = f"reward missing keys: {sorted(missing)}"
        raise ValueError(msg)
    w = reward["weights"]
    if not isinstance(w, dict):
        msg = f"reward.weights must be a mapping, got {type(w).__name__}"
        raise ValueError(msg)
    wm = REQUIRED_REWARD_WEIGHT_KEYS - w.keys()
    if wm:
        msg = f"reward.weights missing keys: {sorted(wm)}"
        raise ValueError(msg)
    for k in REQUIRED_REWARD_WEIGHT_KEYS:
        if not isinstance(w[k], (int, float)):
            msg = f"reward.weights.{k} must be a number"
            raise TypeError(msg)
    if not isinstance(reward["cascade_penalty_scale"], (int, float)):
        msg = "reward.cascade_penalty_scale must be a number"
        raise TypeError(msg)
    thr = reward["cascade_error_rate_threshold"]
    if not isinstance(thr, (int, float)) or not (0.0 <= float(thr) <= 1.0):
        msg = "reward.cascade_error_rate_threshold must be a number between 0 and 1"
        raise ValueError(msg)


def validate_action_config(action: dict[str, Any], policy: dict[str, Any]) -> None:
    """Raise if DQN bins or PPO flags disagree with policy bounds."""
    if not isinstance(action, dict):
        msg = f"action must be a mapping, got {type(action).__name__}"
        raise ValueError(msg)
    missing = REQUIRED_ACTION_KEYS - action.keys()
    if missing:
        msg = f"action missing keys: {sorted(missing)}"
        raise ValueError(msg)
    algo = action["algorithm"]
    if algo not in VALID_ACTION_ALGORITHMS:
        msg = f"action.algorithm must be one of {sorted(VALID_ACTION_ALGORITHMS)}, got {algo!r}"
        raise ValueError(msg)

    r_min = int(policy["retry"]["min"])
    r_max = int(policy["retry"]["max"])
    if r_min > r_max:
        msg = "policy.retry.min must be <= policy.retry.max"
        raise ValueError(msg)

    b_min = float(policy["backoff"]["min_multiplier"])
    b_max = float(policy["backoff"]["max_multiplier"])
    t_min = float(policy["timeout_ms"]["min"])
    t_max = float(policy["timeout_ms"]["max"])

    dqn = action["dqn"]
    if not isinstance(dqn, dict):
        msg = f"action.dqn must be a mapping, got {type(dqn).__name__}"
        raise ValueError(msg)
    bo = dqn.get("backoff_multipliers")
    to = dqn.get("timeout_ms_bins")
    if not isinstance(bo, list) or not bo:
        msg = "action.dqn.backoff_multipliers must be a non-empty list"
        raise ValueError(msg)
    if not isinstance(to, list) or not to:
        msg = "action.dqn.timeout_ms_bins must be a non-empty list"
        raise ValueError(msg)
    for i, x in enumerate(bo):
        if not isinstance(x, (int, float)):
            msg = f"action.dqn.backoff_multipliers[{i}] must be a number"
            raise TypeError(msg)
        xf = float(x)
        if xf < b_min or xf > b_max:
            msg = (
                f"action.dqn.backoff_multipliers[{i}]={xf} is outside "
                f"policy.backoff bounds [{b_min}, {b_max}]"
            )
            raise ValueError(msg)
    prev = None
    for i, x in enumerate(to):
        if not isinstance(x, (int, float)):
            msg = f"action.dqn.timeout_ms_bins[{i}] must be a number"
            raise TypeError(msg)
        xf = float(x)
        if xf < t_min or xf > t_max:
            msg = (
                f"action.dqn.timeout_ms_bins[{i}]={xf} is outside "
                f"policy.timeout_ms bounds [{t_min}, {t_max}]"
            )
            raise ValueError(msg)
        if prev is not None and xf < prev:
            msg = "action.dqn.timeout_ms_bins must be sorted in non-decreasing order"
            raise ValueError(msg)
        prev = xf

    ppo = action["ppo"]
    if not isinstance(ppo, dict):
        msg = f"action.ppo must be a mapping, got {type(ppo).__name__}"
        raise ValueError(msg)
    upb = ppo.get("use_policy_bounds")
    if not isinstance(upb, bool):
        msg = "action.ppo.use_policy_bounds must be a boolean"
        raise TypeError(msg)
    if not upb:
        msg = (
            "action.ppo.use_policy_bounds=false is not supported yet; use true or extend the loader"
        )
        raise ValueError(msg)


def validate_safeguards_config(safeguards: dict[str, Any], policy: dict[str, Any]) -> None:
    """Raise if safeguard bounds disagree with each other or with policy defaults."""
    if not isinstance(safeguards, dict):
        msg = f"safeguards must be a mapping, got {type(safeguards).__name__}"
        raise ValueError(msg)
    missing = REQUIRED_SAFEGUARDS_KEYS - safeguards.keys()
    if missing:
        msg = f"safeguards missing keys: {sorted(missing)}"
        raise ValueError(msg)

    mr = int(safeguards["max_retries"])
    if mr < 0:
        msg = "safeguards.max_retries must be >= 0"
        raise ValueError(msg)
    p_retry_max = int(policy["retry"]["max"])
    if mr != p_retry_max:
        msg = (
            f"safeguards.max_retries ({mr}) should match policy.retry.max ({p_retry_max}) "
            "so training and serving agree"
        )
        raise ValueError(msg)

    bb = safeguards["backoff_multiplier_bounds"]
    if not isinstance(bb, list) or len(bb) != 2:
        msg = "safeguards.backoff_multiplier_bounds must be a list of two numbers [min, max]"
        raise ValueError(msg)
    b_lo, b_hi = float(bb[0]), float(bb[1])
    if b_lo > b_hi:
        msg = "safeguards.backoff_multiplier_bounds: min must be <= max"
        raise ValueError(msg)

    tb = safeguards["timeout_ms_bounds"]
    if not isinstance(tb, list) or len(tb) != 2:
        msg = "safeguards.timeout_ms_bounds must be a list of two numbers [min, max]"
        raise ValueError(msg)
    t_lo, t_hi = float(tb[0]), float(tb[1])
    if t_lo > t_hi:
        msg = "safeguards.timeout_ms_bounds: min must be <= max"
        raise ValueError(msg)

    mpc = int(safeguards["max_policy_changes_per_route_per_minute"])
    if mpc <= 0:
        msg = "safeguards.max_policy_changes_per_route_per_minute must be > 0"
        raise ValueError(msg)

    thr = float(safeguards["circuit_breaker_error_rate_threshold"])
    clr = float(safeguards["circuit_breaker_clear_error_rate_threshold"])
    for name, v in (
        ("circuit_breaker_error_rate_threshold", thr),
        ("circuit_breaker_clear_error_rate_threshold", clr),
    ):
        if not (0.0 <= v <= 1.0):
            msg = f"safeguards.{name} must be between 0 and 1"
            raise ValueError(msg)
    if clr >= thr:
        msg = (
            "safeguards.circuit_breaker_clear_error_rate_threshold must be strictly less than "
            "circuit_breaker_error_rate_threshold (hysteresis)"
        )
        raise ValueError(msg)

    ttl = float(safeguards["route_state_ttl_seconds"])
    if ttl <= 0:
        msg = "safeguards.route_state_ttl_seconds must be > 0"
        raise ValueError(msg)

    pb = policy["backoff"]
    pt = policy["timeout_ms"]
    if b_lo > float(pb["min_multiplier"]) or b_hi < float(pb["max_multiplier"]):
        msg = "safeguards.backoff_multiplier_bounds must cover policy.backoff min/max multipliers"
        raise ValueError(msg)
    if t_lo > float(pt["min"]) or t_hi < float(pt["max"]):
        msg = "safeguards.timeout_ms_bounds must cover policy.timeout_ms min/max"
        raise ValueError(msg)
