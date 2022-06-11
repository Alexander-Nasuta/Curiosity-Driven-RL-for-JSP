import sys

import numpy as np
import torch

from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from jss_rl.sb3.curiosity.icm_wrapper import IntrinsicCuriosityModuleWrapper
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor

if __name__ == '__main__':

    n_envs = 2
    timelimit = 100_000

    size = 7

    desc = [
        "S" + "F" * (size - 1),
        *["F" * size] * (size - 3),
        "F" * (size - 2) + "G" + "F",
        "F" * (size),
    ]

    venv = make_vec_env_without_monitor(
        env_id="FrozenLake-v1",
        env_kwargs={
            "desc": desc,
            "is_slippery": False,
        },
        wrapper_class=TimeLimit,
        wrapper_kwargs={
            "max_episode_steps": (size - 1) * 2
        },
        n_envs=n_envs
    )

    no_icm_venv = VecMonitor(venv=venv)
    no_icm_model = PPO('MlpPolicy', no_icm_venv, verbose=1)

    icm_kwargs = {
        "beta": 0.2,
        "eta": 0.2,
        "lr": 0.001,
        "device": 'cpu',
        "feature_dim": 128,
        "feature_net_hiddens": [],
        "feature_net_activation": torch.nn.ReLU(),
        "inverse_feature_net_hiddens": [256],
        "inverse_feature_net_activation": torch.nn.ReLU(),
        "forward_fcnet_net_hiddens": [256],
        "forward_fcnet_net_activation": torch.nn.ReLU(),
        "postprocess_every_n_steps": 10,
        "postprocess_sample_size": 100,
        "memory_capacity": 1_000,
        "shuffle_memory_samples": True,
        "clear_memory_on_reset": False,
    }

    venv = make_vec_env_without_monitor(
        env_id="FrozenLake-v1",
        env_kwargs={
            "desc": desc,
            "is_slippery": False,
        },
        wrapper_class=TimeLimit,
        wrapper_kwargs={
            "max_episode_steps": (size - 1) * 2
        },
        n_envs=n_envs
    )

    # icm_venv = venv
    icm_venv = IntrinsicCuriosityModuleWrapper(venv=venv, **icm_kwargs)
    icm_venv = VecMonitor(venv=icm_venv)
    icm_model = PPO('MlpPolicy', icm_venv, verbose=1)


    print("benchmarking icm...")
    steps = 0
    obs = icm_venv.reset()
    goal_reached = False
    while not goal_reached:
        steps += 1
        action, _ = icm_model.predict(observation=obs, deterministic=False)
        obs, rewards, dones, infos = icm_venv.step(action)
        for r, info in zip(rewards, infos):
            if info['extrinsic_reward'] > 0.0:
                goal_reached = True
                print(f"reached goal in {steps} timesteps.")

        if steps > timelimit:
            print("timelimit exceeded.")
            break


    print("benchmarking no icm...")
    steps = 0
    obs = no_icm_venv.reset()
    goal_reached = False
    while not goal_reached:
        action, _ = no_icm_model.predict(observation=obs, deterministic=False)
        obs, rewards, dones, infos = no_icm_venv.step(action)
        for r in rewards:
            if r > 0:
                goal_reached = True
                print(f"reached goal in {steps} timesteps.")

        if steps > timelimit:
            print("timelimit exceeded.")
            break

        steps += 1
