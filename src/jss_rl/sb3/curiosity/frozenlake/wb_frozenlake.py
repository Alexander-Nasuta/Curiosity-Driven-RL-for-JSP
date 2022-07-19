import sys
import numpy as np
from collections import deque
from types import ModuleType
from typing import List

import torch as th
import wandb as wb
import stable_baselines3 as sb3

from gym.wrappers import TimeLimit
from rich.progress import track
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs
from wandb.integration.sb3 import WandbCallback
from statistics import mean

from jss_rl.sb3.curiosity.curiosity_info_wrapper import CuriosityInfoWrapper
from jss_rl.sb3.curiosity.icm import IntrinsicCuriosityModuleWrapper
from jss_rl.sb3.curiosity.ec import EpisodicCuriosityModuleWrapper
from jss_rl.sb3.util.callbacks.episode_end_moving_average_rollout_end_logger_callback import \
    EpisodeEndMovingAverageRolloutEndLoggerCallback
from jss_rl.sb3.util.callbacks.wb_info_logger_callback import WB_InfoLoggerCallback
from jss_rl.sb3.util.info_field_moving_avarege_logger_callback import InfoFieldMovingAverageLogger
from jss_utils import PATHS
from jss_utils.jss_logger import log
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_rl.sb3.util.moving_avarage import MovingAverage


class DistanceWrapper(VecEnvWrapper):

    def __init__(self, venv):
        self.distances = MovingAverage(capacity=1000)
        VecEnvWrapper.__init__(self, venv=venv)

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()

        for i, o in enumerate(observations):
            x, y = o % 8, o // 8  # frozen lake with 8x8 size
            infos[i]["distance_from_origin"] = (x ** 2 + y ** 2) ** 0.5

        return observations, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        observations = self.venv.reset()
        return observations


class FrozenlakeLoggerCallBack(BaseCallback):

    def __init__(self, wandb_ref: ModuleType = wb, verbose=0):
        super(FrozenlakeLoggerCallBack, self).__init__(verbose)

        self.wandb_ref = wandb_ref

        self.max_distance = 0.0
        self.visited_states = set()
        self.visited_state_action_pairs = set()
        self.max_return = 0.0

        self.temp_trajectories = [
            deque(maxlen=3 * 16) for _ in range(8)  # num envs
        ]
        self.best_trajectory = None

        self._n_rollouts = 0

        self.log_fields = [
            "extrinsic_return",
            "intrinsic_return",
            "bonus_return",
            "total_return",
            "n_postprocessings",
            "n_total_episodes",
            "_num_timesteps",
            "distance_from_origin",

            "loss",
            "inverse_loss",
            "forward_loss",
        ]

    def _get_vals(self, field: str) -> List:
        return [env_info[field] for env_info in self.locals['infos'] if field in env_info.keys()]

    def _on_step(self) -> bool:
        # self.num_timesteps
        for i, obs in enumerate(self.locals["obs_tensor"]):
            self.temp_trajectories[i].append(obs)




        self.max_distance = max(self.max_distance, *self._get_vals("distance_from_origin"))

        new_max = max(0.0, self.max_return, *self._get_vals("total_return"))
        if new_max > self.max_return:
            self.max_return = new_max
            max_i = np.argmax(np.array(self._get_vals("distance_from_origin")))
            self.best_trajectory = list(self.temp_trajectories[max_i])

        self.visited_states = self.visited_states.union([obs.item() for obs in self.locals["obs_tensor"]])

        self.visited_state_action_pairs = self.visited_state_action_pairs.union(
            [(obs.item(), actions) for obs, actions in zip(self.locals["obs_tensor"], self.locals["actions"])]
        )

        # note num_timesteps increments always in num_envs (here 8)
        if self.num_timesteps and self.num_timesteps % 10_000 == 0:
            tab = self.wandb_ref.Table(
                columns=[" ", *[f"step_{i}" for i in range(len(self.temp_trajectories[0]))]],
                data=[
                    ["state", *[elem.item() for elem in self.temp_trajectories[0]]],
                    ["state (index 1)", *[elem.item() + 1 for elem in self.temp_trajectories[0]]],
                    ["col", *[elem.item() % 8 for elem in self.temp_trajectories[0]]],
                    ["row", *[elem.item() // 8 for elem in self.temp_trajectories[0]]],
                ]
            )
            self.wandb_ref.log({
                f"trajectory_{self.num_timesteps}steps_env_0": tab,
                "num_timesteps": self.num_timesteps
            })

        if self.wandb_ref:
            self.wandb_ref.log({
                **{f: mean(self._get_vals(f)) for f in self.log_fields if self._get_vals(f)},
                "max_distance": self.max_distance,
                "n_visited_states": len(self.visited_states),
                "n_visited_state_action_pairs": len(self.visited_state_action_pairs),
                "explored_states": len(self.visited_states) / (8 * 8),
                # no actions in terminal state
                "explored_state_action_pairs": len(self.visited_state_action_pairs) / (8 * 8 * 4 - 4),
                "num_timesteps": self.num_timesteps
            })

        return True

    def _on_rollout_end(self) -> None:
        self._n_rollouts += 1
        tab = self.wandb_ref.Table(
            columns=[" ", *[f"step_{i}" for i in range(len(self.temp_trajectories[0]))]],
            data=[
                ["state", *[elem.item() for elem in self.temp_trajectories[0]]],
                ["state (index 1)", *[elem.item() + 1 for elem in self.temp_trajectories[0]]],
                ["col", *[elem.item() % 8 for elem in self.temp_trajectories[0]]],
                ["row", *[elem.item() // 8 for elem in self.temp_trajectories[0]]],
            ]
        )
        self.wandb_ref.log({
            f"rollout_{self._n_rollouts}_sample_trajectory": tab,
            "num_timesteps": self.num_timesteps
        })

    def _on_training_end(self) -> None:
        tab = self.wandb_ref.Table(
            columns=[" ", *[f"step_{i}" for i in range(len(self.best_trajectory))]],
            data=[
                ["state", *[elem.item() for elem in self.best_trajectory]],
                ["state (index 1)", *[elem.item() + 1 for elem in self.best_trajectory]],
                ["col", *[elem.item() % 8 for elem in self.best_trajectory]],
                ["row", *[elem.item() // 8 for elem in self.best_trajectory]],
            ]
        )
        self.wandb_ref.log({
            "highest_score_trajectory": tab,
            "num_timesteps": self.num_timesteps
        })


def main(num_samples=1):
    project = "frozenlake-sb3_dev"
    config = {}
    config["total_timesteps"] = 50_000
    config["env_name"] = "FrozenLake-v1"
    config["env_kwargs"] = {
        "desc": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFG",
        ],
        "is_slippery": False,
    }
    config["wrapper_class"] = TimeLimit
    config["wrapper_kwargs"] = {"max_episode_steps": 16}  # basically the same as config["horizon"] = 16
    config["n_envs"] = 6
    config["model_policy"] = "MlpPolicy"
    config["model_hyper_parameters"] = {
        "gamma": 0.99,  # discount factor,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "clip_range": 0.541,
        "clip_range_vf": 26,
        "ent_coef": 0.0,
        "normalize_advantage": True,
        # "target_kl": 0.05047, # for early stopping
        "policy_kwargs": {
            "net_arch": [{
                "pi": [64, 64],
                "vf": [64, 64],
            }],
            "ortho_init": True,
            "activation_fn": th.nn.Tanh,  # th.nn.ReLU
            "optimizer_kwargs": {  # for th.optim.Adam
                "eps": 1e-5
            }
        }
    }

    for _ in track(range(num_samples), description="running experiments with plain PPO"):
        break
        run = wb.init(
            project=project,
            group="PPO",
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

        logger_cb = FrozenlakeLoggerCallBack()

        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[wb_cb, logger_cb]
        )

        run.finish()

    for _ in track(range(num_samples), description="running experiments with PPO and ICM"):
        break
        icm_config = config.copy()
        icm_config["IntrinsicCuriosityModuleWrapper"] = {
            "beta": 0.2,
            "eta": 1.0,
            "lr": 0.001,
            "device": 'cpu',
            "feature_dim": 288,
            "feature_net_hiddens": [],
            "feature_net_activation": "relu",
            "inverse_feature_net_hiddens": [256],
            "inverse_feature_net_activation": "relu",
            "forward_fcnet_net_hiddens": [256],
            "forward_fcnet_net_activation": "relu",

            "maximum_sample_size": 16,

            "clear_memory_on_end_of_episode": True,
            "postprocess_on_end_of_episode": True,

            "clear_memory_every_n_steps": None,
            "postprocess_every_n_steps": None

        }

        run = wb.init(
            project=project,
            group="PPO + ICM",
            config=icm_config,
            sync_tensorboard=True,
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            dir=f"{PATHS.WANDB_PATH}/",
        )

        venv = make_vec_env_without_monitor(
            env_id=icm_config["env_name"],
            env_kwargs=icm_config["env_kwargs"],
            wrapper_class=icm_config["wrapper_class"],
            wrapper_kwargs=icm_config["wrapper_kwargs"],
            n_envs=icm_config["n_envs"]  # basically the same as config["num_workers"] = 0
        )

        venv = DistanceWrapper(venv=venv)  # equivalent to `MyCallBack`

        venv = IntrinsicCuriosityModuleWrapper(
            venv=venv,
            **icm_config["IntrinsicCuriosityModuleWrapper"]
        )

        venv = VecMonitor(venv=venv)

        model = sb3.PPO(
            "MlpPolicy",
            env=venv,
            verbose=1,
            tensorboard_log=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
            **icm_config["model_hyper_parameters"]
        )

        wb_cb = WandbCallback(
            gradient_save_freq=100,
            model_save_path=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
            verbose=1,
        )

        logger_cb = FrozenlakeLoggerCallBack(
            wandb_ref=wb,
        )

        model.learn(
            total_timesteps=icm_config["total_timesteps"],
            callback=[wb_cb, logger_cb]
        )

        run.finish()

    for _ in track(range(num_samples), description="running experiments with PPO and EC"):
        ec_config = config.copy()
        ec_config["EpisodicCuriosityModuleWrapper"] = {
            "alpha": 1.0,
            "beta": 0.5,
            "gamma": 2,
            "embedding_dim": 64,
            "episodic_memory_capacity": 8,
            "clear_memory_every_episode": True,
            "exploration_steps": 20_000,
        }

        run = wb.init(
            project=project,
            group="PPO + EC (no memory reset)"
            if not ec_config["EpisodicCuriosityModuleWrapper"]["clear_memory_every_episode"]
            else "PPO + EC",
            config=ec_config,
            sync_tensorboard=True,
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            dir=f"{PATHS.WANDB_PATH}/",
        )

        venv = make_vec_env_without_monitor(
            env_id=ec_config["env_name"],
            env_kwargs=ec_config["env_kwargs"],
            wrapper_class=ec_config["wrapper_class"],
            wrapper_kwargs=ec_config["wrapper_kwargs"],
            n_envs=ec_config["n_envs"]  # basically the same as config["num_workers"] = 0
        )

        venv = DistanceWrapper(venv=venv)  # equivalent to `MyCallBack`

        venv = EpisodicCuriosityModuleWrapper(
            venv=venv,
            **ec_config["EpisodicCuriosityModuleWrapper"]
        )

        venv = VecMonitor(venv=venv)

        model = sb3.PPO(
            "MlpPolicy",
            env=venv,
            verbose=1,
            tensorboard_log=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
            **ec_config["model_hyper_parameters"]
        )

        wb_cb = WandbCallback(
            gradient_save_freq=100,
            model_save_path=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
            verbose=1,
        )

        logger_cb = FrozenlakeLoggerCallBack(
            wandb_ref=wb,
        )

        model.learn(
            total_timesteps=ec_config["total_timesteps"],
            callback=[wb_cb, logger_cb]
        )

        run.finish()


if __name__ == '__main__':
    main(num_samples=1)
