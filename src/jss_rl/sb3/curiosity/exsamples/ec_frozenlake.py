import sys

import numpy as np
import torch

from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from jss_rl.sb3.curiosity.ec_wrapper import EpisodicCuriosityEnvWrapper
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor

if __name__ == '__main__':

    n_envs = 2
    timelimit = 100_000

    size = 8

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

    no_ec_venv = VecMonitor(venv=venv)
    no_ec_model = PPO('MlpPolicy', no_ec_venv, verbose=1)

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

    ec_kwargs = {

    }

    # icm_venv = venv
    ec_venv = EpisodicCuriosityEnvWrapper(venv=venv)
    ec_venv = VecMonitor(venv=ec_venv)
    ec_model = PPO('MlpPolicy', ec_venv, verbose=1)


    print("benchmarking ec...")
    steps = 0
    obs = ec_venv.reset()
    goal_reached = False
    while not goal_reached:
        steps += 1
        action, _ = ec_model.predict(observation=obs, deterministic=False)
        obs, rewards, dones, infos = ec_venv.step(action)
        for r, info in zip(rewards, infos):
            if info['extrinsic_reward'] > 0.0:
                goal_reached = True
                print(f"reached goal in {steps} timesteps.")

        if steps > timelimit:
            print("timelimit exceeded.")
            break


    print("benchmarking no ec...")
    steps = 0
    obs = no_ec_venv.reset()
    goal_reached = False
    while not goal_reached:
        action, _ = no_ec_model.predict(observation=obs, deterministic=False)
        obs, rewards, dones, infos = no_ec_venv.step(action)
        for r in rewards:
            if r > 0:
                goal_reached = True
                print(f"reached goal in {steps} timesteps.")

        if steps > timelimit:
            print("timelimit exceeded.")
            break

        steps += 1
