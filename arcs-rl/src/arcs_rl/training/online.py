"""
Learn while interacting with the simulator: DQN or PPO with Stable-Baselines3.

DQN uses a wrapper so the library sees one Discrete action instead of MultiDiscrete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from arcs_rl.config import effective_training_seed
from arcs_rl.monitoring import maybe_start_training_metrics
from arcs_rl.training._device import resolve_training_device
from arcs_rl.training.metrics_callback import TrainingMetricsCallback
from arcs_rl.training.vec_env import make_vec_env


def run_online_training(
    config: dict[str, Any],
    *,
    run_name: str = "online_run",
) -> Path:
    """
    Run Stable-Baselines3 learning on the mock microservice environment.

    Saves checkpoints under training.checkpoint_dir / run_name and returns the final model path.
    """
    seed = effective_training_seed(config)
    tcfg = config["training"]
    device = resolve_training_device(str(tcfg["device"]))
    set_random_seed(seed, using_cuda=device == "cuda")

    vec = make_vec_env(config)
    algo = str(config["action"]["algorithm"]).lower()
    out_dir = Path(tcfg["checkpoint_dir"]).expanduser() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = str(out_dir / "ckpt")
    callbacks: list[Any] = [
        CheckpointCallback(
            save_freq=max(1, int(tcfg["online"]["total_timesteps"]) // 10),
            save_path=save_path,
            name_prefix=f"{algo}_",
        )
    ]
    metrics = maybe_start_training_metrics(config)
    if metrics is not None:
        callbacks.append(TrainingMetricsCallback(metrics, algorithm=algo))

    total = int(tcfg["online"]["total_timesteps"])
    lr = float(tcfg["learning_rate"])

    if algo == "dqn":
        dqn_cfg = tcfg["dqn"]
        dqn_model = DQN(
            "MlpPolicy",
            vec,
            learning_rate=lr,
            buffer_size=int(dqn_cfg["buffer_size"]),
            exploration_fraction=float(dqn_cfg["exploration_fraction"]),
            exploration_initial_eps=float(dqn_cfg["exploration_initial_eps"]),
            exploration_final_eps=float(dqn_cfg["exploration_final_eps"]),
            verbose=1,
            seed=seed,
            device=device,
        )
        dqn_model.learn(total_timesteps=total, callback=callbacks)
        final_path = out_dir / f"{algo}_online.zip"
        dqn_model.save(str(final_path))
        return final_path

    if algo == "ppo":
        ppo_cfg = tcfg["ppo"]
        ppo_model = PPO(
            "MlpPolicy",
            vec,
            learning_rate=lr,
            n_steps=int(ppo_cfg["n_steps"]),
            batch_size=int(ppo_cfg["batch_size"]),
            verbose=1,
            seed=seed,
            device=device,
        )
        ppo_model.learn(total_timesteps=total, callback=callbacks)
        final_path = out_dir / f"{algo}_online.zip"
        ppo_model.save(str(final_path))
        return final_path

    msg = f"Unsupported action.algorithm for online training: {algo!r}"
    raise ValueError(msg)
