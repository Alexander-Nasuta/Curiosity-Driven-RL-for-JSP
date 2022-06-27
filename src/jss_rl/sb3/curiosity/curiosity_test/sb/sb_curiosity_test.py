import pprint
import torch
import copy

import numpy as np

from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from jss_rl.sb3.curiosity.ec_wrapper import EpisodicCuriosityEnvWrapper
from jss_rl.sb3.curiosity.icm_wrapper import IntrinsicCuriosityModuleWrapper
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_rl.sb3.util.moving_avarage import MovingAverage
from jss_utils.jss_logger import log


class DistanceWrapper(VecEnvWrapper):

    def __init__(self, venv, log_interval=840):
        self.distances = MovingAverage(capacity=1000)
        self._n_step = 0
        self.log_iterval = log_interval  # Highly composite number
        VecEnvWrapper.__init__(self, venv=venv)

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        self._n_step += self.venv.num_envs
        observations, rewards, dones, infos = self.venv.step_wait()

        for i, o in enumerate(observations):
            x, y = o % 8, o // 8  # frozen lake with 8x8 size
            self.distances.add((x ** 2 + y ** 2) ** 0.5)

        if self._n_step % self.log_iterval == 0:
            log.info(f"mean distance from origin = {self.distances.mean():.4f}")

        return observations, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        observations = self.venv.reset()
        return observations


class ObsSpaceWrapper(VecEnvWrapper):

    def __init__(self, venv):
        obs_shape = venv.observation_space.shape
        VecEnvWrapper.__init__(self, venv=venv)

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()

        print(observations)
        if self.obs_shape_type == "empty_tupel":
            print("hey")
            new_obs = np.array([np.ravel(o) for o in observations])
            print(new_obs.shape)
            return new_obs, rewards, dones, infos

        return observations, rewards, dones, infos

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        observations = self.venv.reset()
        return observations


def test_curiosity_on_frozen_lake():
    log.info("comparing icm performance with ppo algorithm on 'FrozenLake-v1' enviorment")

    env_name = "FrozenLake-v1"
    env_kwargs = {
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
    icm_kwargs = {
        "beta": 0.2,
        "eta": 1.0,
        "lr": 0.001,
        "device": 'cpu',
        "feature_dim": 288,
        "feature_net_hiddens": [],
        "feature_net_activation": torch.nn.ReLU(),
        "inverse_feature_net_hiddens": [256],
        "inverse_feature_net_activation": torch.nn.ReLU(),
        "forward_fcnet_net_hiddens": [256],
        "forward_fcnet_net_activation": torch.nn.ReLU(),
        "postprocess_every_n_steps": 100,
        "postprocess_sample_size": 100,
        "memory_capacity": 10_000,
        "shuffle_memory_samples": True,
        "clear_memory_on_reset": True,
        "exploration_steps": 0,
    }
    ec_kwargs = {

    }

    venv = make_vec_env_without_monitor(
        env_id=env_name,
        env_kwargs=env_kwargs,
        wrapper_class=TimeLimit,
        wrapper_kwargs={"max_episode_steps": 16},  # basically the same as config["horizon"] = 16
        n_envs=1  # basically the same as config["num_workers"] = 0
    )
    venv = DistanceWrapper(venv=venv)  # equivalent to `MyCallBack`

    no_icm_venv = VecMonitor(venv=venv)
    no_icm_model = PPO('MlpPolicy', no_icm_venv, verbose=0)

    venv = make_vec_env_without_monitor(
        env_id=env_name,
        env_kwargs=env_kwargs,
        wrapper_class=TimeLimit,
        wrapper_kwargs={"max_episode_steps": 16},  # basically the same as config["horizon"] = 16
        n_envs=1  # basically the same as config["num_workers"] = 0
    )
    venv = DistanceWrapper(venv=venv)  # equivalent to `MyCallBack`

    icm_venv = IntrinsicCuriosityModuleWrapper(
        venv=venv,
        **icm_kwargs
    )
    icm_model = PPO('MlpPolicy', icm_venv, verbose=0)

    ec_venv = make_vec_env_without_monitor(
        env_id=env_name,
        env_kwargs=env_kwargs,
        wrapper_class=TimeLimit,
        wrapper_kwargs={"max_episode_steps": 16},  # basically the same as config["horizon"] = 16
        n_envs=1  # basically the same as config["num_workers"] = 0
    )
    ec_venv = DistanceWrapper(venv=ec_venv)  # equivalent to `MyCallBack`
    ec_venv = ObsSpaceWrapper(venv=ec_venv)
    ec_venv = EpisodicCuriosityEnvWrapper(venv=venv, **ec_kwargs)
    ec_model = PPO('MlpPolicy', ec_venv, verbose=0)

    budget = 25_000
    log.info("evaluating performance without icm")
    no_icm_model.learn(total_timesteps=budget)

    log.info("evaluating performance with icm")
    icm_model.learn(total_timesteps=budget)

    log.info("evaluating performance with ec")
    # ec_model.learn(total_timesteps=budget)



def test_curiosity_on_partially_observable_domain():
    pass


if __name__ == '__main__':
    test_curiosity_on_frozen_lake()
