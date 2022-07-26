from collections import deque
from statistics import mean
from types import ModuleType
from typing import Dict, List

import gym
import torch as th
import wandb as wb
import gym_minigrid
import numpy as np
import stable_baselines3 as sb3
from gym.wrappers import TimeLimit
from ray.rllib.utils import one_hot
from rich.progress import track
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

from jss_rl.sb3.curiosity_modules.curiosity_info_wrapper import CuriosityInfoWrapper
from jss_rl.sb3.curiosity.ec import EpisodicCuriosityModuleWrapper
from jss_rl.sb3.curiosity.icm import IntrinsicCuriosityModuleWrapper
from jss_utils import PATHS


class OneHotWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, vector_index, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.single_frame_dim = 49 * (11 + 6 + 3) + 4
        self.init_x = None
        self.init_y = None
        self.x_positions = []
        self.y_positions = []
        self.x_y_delta_buffer = deque(maxlen=100)
        self.vector_index = vector_index
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.single_frame_dim,)))

        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(self.single_frame_dim * self.framestack,), dtype=np.float32
        )

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info["100-average dist travelled"] = np.mean(self.x_y_delta_buffer)
        return self.observation(observation), reward, done, info

    def observation(self, obs):
        # Debug output: max-x/y positions to watch exploration progress.
        if self.step_count == 0:
            for _ in range(self.framestack):
                self.frame_buffer.append(np.zeros((self.single_frame_dim,)))
            if self.vector_index == 0:
                if self.x_positions:
                    max_diff = max(
                        np.sqrt(
                            (np.array(self.x_positions) - self.init_x) ** 2
                            + (np.array(self.y_positions) - self.init_y) ** 2
                        )
                    )
                    self.x_y_delta_buffer.append(max_diff)
                    self.x_positions = []
                    self.y_positions = []
                self.init_x = self.agent_pos[0]
                self.init_y = self.agent_pos[1]

        # Are we carrying the key?
        # if self.carrying is not None:
        #    print("Carrying KEY!!")

        self.x_positions.append(self.agent_pos[0])
        self.y_positions.append(self.agent_pos[1])

        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(obs[:, :, 0], depth=11)
        colors = one_hot(obs[:, :, 1], depth=6)
        states = one_hot(obs[:, :, 2], depth=3)
        # Is the door we see open?
        # for x in range(7):
        #    for y in range(7):
        #        if objects[x, y, 4] == 1.0 and states[x, y, 0] == 1.0:
        #            print("Door OPEN!!")

        all_ = np.concatenate([objects, colors, states], -1)
        all_flat = np.reshape(all_, (-1,))
        direction = one_hot(np.array(self.agent_dir), depth=4).astype(np.float32)
        single_frame = np.concatenate([all_flat, direction])
        self.frame_buffer.append(single_frame)
        return np.concatenate(self.frame_buffer)


def env_maker(config: Dict, wrapper_class=None, wrapper_kwargs=None):
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    name = config.get("name", "MiniGrid-Empty-5x5-v0")
    framestack = config.get("framestack", 4)
    env = gym.make(name)
    # Only use image portion of observation (discard goal and direction).
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    env = OneHotWrapper(
        env,
        config.vector_index if hasattr(config, "vector_index") else 0,
        framestack=framestack,
    )
    if wrapper_class is not None:
        env = wrapper_class(env, **wrapper_kwargs)
    return env


class MinigridLoggerCallBack(BaseCallback):

    def __init__(self, wandb_ref: ModuleType = wb, verbose=0):
        super(MinigridLoggerCallBack, self).__init__(verbose)

        self.wandb_ref = wandb_ref

        self.max_distance = 0.0
        self.visited_states = set()
        self.visited_state_action_pairs = set()

        self.log_fields = [
            "extrinsic_return",
            "intrinsic_return",
            "bonus_return",
            "total_return",
            "n_postprocessings",
            "n_total_episodes",
            "_num_timesteps",

            "loss",
            "inverse_loss",
            "forward_loss",

            "100-average dist travelled",
            "memory_size"
        ]

    def _get_vals(self, field: str) -> List:
        return [env_info[field] for env_info in self.locals['infos'] if field in env_info.keys()]

    def _on_step(self) -> bool:
        self.max_distance = 0

        self.visited_states = self.visited_states.union([tuple(obs.tolist()) for obs in self.locals["obs_tensor"]])

        self.visited_state_action_pairs = self.visited_state_action_pairs.union(
            [(tuple(obs.tolist()), actions) for obs, actions in zip(self.locals["obs_tensor"], self.locals["actions"])]
        )

        if self.wandb_ref:
            self.wandb_ref.log({
                **{f: mean(self._get_vals(f)) for f in self.log_fields if self._get_vals(f)},
                "n_visited_states": len(self.visited_states),
                "n_visited_state_action_pairs": len(self.visited_state_action_pairs),
                "num_timesteps": self.num_timesteps
            })

        return True


def main(num_samples=5):
    project = "minigrid-sb3-dev"

    config = {}
    config["total_timesteps"] = 30_000
    config["env_config"] = {
        # Also works with:
        # - MiniGrid-MultiRoom-N4-S5-v0
        # - MiniGrid-MultiRoom-N2-S4-v0
        "name": "MiniGrid-Empty-8x8-v0",
        "framestack": 1,  # seems to work even w/o framestacking
    }
    config["wrapper_class"] = TimeLimit
    config["wrapper_kwargs"] = {"max_episode_steps": 15}
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
                "pi": [256, 256],
                "vf": [256, 256],
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

        venv = DummyVecEnv([
            lambda: env_maker(
                config=config["env_config"],
                wrapper_class=config["wrapper_class"],
                wrapper_kwargs=config["wrapper_kwargs"])
        ])

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

        logger_cb = MinigridLoggerCallBack(
            wandb_ref=wb
        )

        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[wb_cb, logger_cb]
        )

        run.finish()

    for _ in track(range(num_samples), description="running experiments with plain PPO + ICM"):
        break
        icm_config = config.copy()

        icm_config["IntrinsicCuriosityModuleWrapper"] = {
            "beta": 0.2,
            "eta": 0.1,
            "lr": 0.0003,
            "device": 'cpu',
            "feature_dim": 64,
            "feature_net_hiddens": [],
            "feature_net_activation": "relu",
            "inverse_feature_net_hiddens": [256],
            "inverse_feature_net_activation": "relu",
            "forward_fcnet_net_hiddens": [256],
            "forward_fcnet_net_activation": "relu",

            # "maximum_sample_size": 16,

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

        venv = DummyVecEnv([
            lambda: env_maker(
                config=icm_config["env_config"],
                wrapper_class=icm_config["wrapper_class"],
                wrapper_kwargs=icm_config["wrapper_kwargs"])
        ])

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

        logger_cb = MinigridLoggerCallBack()

        model.learn(
            total_timesteps=icm_config["total_timesteps"],
            callback=[wb_cb, logger_cb]
        )

        run.finish()

    for _ in track(range(num_samples), description="running experiments with plain PPO + EC"):
        ec_config = config.copy()

        ec_config["EpisodicCuriosityModuleWrapper"] = {
            "alpha": 0.1,
            "beta": 0.5,
            "lr": 1e-3,
            "gamma": 2,
            "ec_capacity": 20,
        }

        run = wb.init(
            project=project,
            group="PPO + EC",
            config=ec_config,
            sync_tensorboard=True,
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
            dir=f"{PATHS.WANDB_PATH}/",
        )

        venv = DummyVecEnv([
            lambda: env_maker(
                config=ec_config["env_config"],
                wrapper_class=ec_config["wrapper_class"],
                wrapper_kwargs=ec_config["wrapper_kwargs"])
        ])

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

        logger_cb = MinigridLoggerCallBack()

        model.learn(
            total_timesteps=ec_config["total_timesteps"],
            callback=[wb_cb, logger_cb]
        )

        run.finish()


if __name__ == '__main__':
    main(num_samples=10)
