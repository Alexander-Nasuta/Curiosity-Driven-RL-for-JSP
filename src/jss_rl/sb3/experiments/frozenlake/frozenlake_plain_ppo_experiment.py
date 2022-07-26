import stable_baselines3 as sb3
import wandb as wb
from stable_baselines3.common.vec_env import VecMonitor
from wandb.integration.sb3 import WandbCallback

import jss_utils.PATHS as PATHS
from jss_rl.sb3.curiosity_modules.curiosity_info_wrapper import CuriosityInfoWrapper

from jss_rl.sb3.experiments.frozenlake.frozenlake_default_config import frozenlake_default_config
from jss_rl.sb3.experiments.frozenlake.frozenlake_distance_wrapper import DistanceWrapper
from jss_rl.sb3.experiments.frozenlake.frozenlake_logger_cb import FrozenlakeLoggerCallBack
from jss_rl.sb3.make_vec_env_without_monitor import make_vec_env_without_monitor


def run_ppo_frozenlake_experiment(total_timesteps: int, *, project: str, group="PPO", additional_config=None):
    if additional_config is None:
        additional_config = {}
    config = frozenlake_default_config.copy()
    config = {
        **config,
        **additional_config,
        "total_timesteps": total_timesteps,
        "curiosity_module": "None",
        "agent_algorithm": "PPO",
        "group_arg": group,
    }

    run = wb.init(
        project=project,
        group=group,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        dir=f"{PATHS.WANDB_PATH}/",
    )

    venv = make_vec_env_without_monitor(
        env_id=config["env_name"],
        env_kwargs=config["env_kwargs"],
        wrapper_class=config["wrapper_class"],
        wrapper_kwargs=config["wrapper_kwargs"],
        n_envs=config["n_envs"]  # basically the same as config["num_workers"] = 0
    )

    venv = DistanceWrapper(venv=venv)
    venv = CuriosityInfoWrapper(venv=venv)
    venv = VecMonitor(venv=venv)

    model = sb3.PPO(
        "MlpPolicy",
        env=venv,
        verbose=1,
        tensorboard_log=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
        **config["model_hyper_parameters"]
    )

    wb_cb = WandbCallback(
        gradient_save_freq=100,
        model_save_path=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
        verbose=1,
    )

    logger_cb = FrozenlakeLoggerCallBack(
        num_envs=config["n_envs"],
        wandb_ref=wb
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wb_cb, logger_cb]
    )

    run.finish()


if __name__ == '__main__':
    wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))
    run_ppo_frozenlake_experiment(
        project="test",
        total_timesteps=50_000,
    )
