import pprint

import gym
import numpy as np
import sb3_contrib
import torch as th
import wandb as wb

import jss_utils.PATHS as PATHS
import jss_utils.jsp_env_utils as env_utils

from types import ModuleType
from rich.progress import track

from wandb.integration.sb3 import WandbCallback

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder

from jss_rl.sb3.curiosity.curiosity_info_wrapper import CuriosityInfoWrapper
from jss_rl.sb3.curiosity.icm2 import IntrinsicCuriosityModuleWrapper
from jss_rl.sb3.curiosity.jss.jss_logger_callback import JssLoggerCallback
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_utils.jss_logger import log

wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))

gym.envs.register(
    id='GraphJsp-v0',
    entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
    kwargs={},
)

PROJECT = "6x6_ft06_tuning"

SWEEP_CONFIG = {
    'method': 'random',
    'metric': {
        'name': 'mean_makespan',
        'goal': 'minimize'
    },
    'parameters': {
        "gamma": {
            "distribution": "uniform",
            "min": 0.9,
            "max": 1,
        },
        "gae_lambda": {
            "distribution": "uniform",
            "min": 0.9,
            "max": 1,
        },
        "learning_rate": {
            "values": [
                1e-2, 3e-2,
                1e-3, 3e-3,
                1e-4, 3e-4,
                1e-5, 3e-5]
        },
        "batch_size": {
            'values': [16, 32, 64, 128, 256, 512, 1024]
        },
        "clip_range": {
            'values': [0.02, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        },
        "clip_range_vf": {
            'values': [26]  # mistake should be between 0.0 and 1.0
        },
        "ent_coef": {
            "values": [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
        },
        "normalize_advantage": {
            'values': [True, False]
        },
        "net_arch_n_layers": {
            'values': [1, 2, 3, 4]
        },
        "net_arch_n_size": {
            'values': [8, 16, 32, 64, 128, 256, 512]
        },
        "ortho_init": {
            'values': [True, False]
        },
        "activation_fn": {
            "values": [
                "Tanh",  # th.nn.Tanh,
                "ReLu",  # th.nn.ReLU
            ]
        },
        "optimizer_eps": {  # for th.optim.Adam
            "values": [1e-5, 1e-6, 1e-7, 1e-8]
        },

        # env params
        "action_mode": {
            'values': ['task', 'job']
        },
        "normalize_observation_space": {
            'values': [True, False]
        },
        "flat_observation_space": {
            'values': [True, False]
        },
        "perform_left_shift_if_possible": {
            'values': [False, True]
        },
        "dtype": {
            'values': ["float32"]
        },

        # eval params
        "n_eval_episodes": {
            'value': 50
        }
    }
}


def perform_run():
    instance_name = "ft06"
    jsp_instance, jsp_instance_details = env_utils.get_benchmark_instance_and_details(name=instance_name)
    _, n_jobs, n_machines = jsp_instance.shape

    RUN_CONFIG = {
        "total_timesteps": 100_000,
        "n_envs": 8,

        "instance_name": instance_name,
        "instance_details": jsp_instance_details,

        "policy_type": MaskableActorCriticPolicy,
        "model_hyper_parameters": {
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
        },

        "env_name": "GraphJsp-v0",
        "env_kwargs": {
            "scale_reward": True,
            "normalize_observation_space": True,
            "flat_observation_space": True,
            "perform_left_shift_if_possible": True,
            "default_visualisations": [
                "gantt_window",
                # "graph_window",  # very expensive
            ]
        },

    }

    with wb.init(
            sync_tensorboard=False,
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
            dir=f"{PATHS.WANDB_PATH}/") as run:

        log.info(f"run name: {run.name}, run id: {run.id}")

        sweep_params = wb.config
        log.info(f"hyper params: {pprint.pformat(sweep_params)}")

        # override run config
        model_params = [
            "gamma",
            "batch_size",
            "clip_range",
            "clip_range_vf",
            "ent_coef",
            "normalize_advantage"
        ]
        for m_param in model_params:
            RUN_CONFIG["model_hyper_parameters"][m_param] = sweep_params[m_param]

        env_params = [
            "normalize_observation_space",
            "flat_observation_space",
            "perform_left_shift_if_possible",
            "dtype",
            "action_mode",
        ]

        for env_param in env_params:
            RUN_CONFIG["env_kwargs"][env_param] = sweep_params[env_param]

        policy_params = [
            "ortho_init",
        ]
        for p_param in policy_params:
            RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"][p_param] = sweep_params[p_param]

        net_arch = [{
            "pi": [sweep_params["net_arch_n_size"]] * sweep_params["net_arch_n_layers"],
            "vf": [sweep_params["net_arch_n_size"]] * sweep_params["net_arch_n_layers"],
        }]
        activation_fn = th.nn.ReLU if sweep_params["activation_fn"] == 'ReLu' else th.nn.Tanh

        RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"]["net_arch"] = net_arch
        RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"]["activation_fn"] = activation_fn
        RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"]["optimizer_kwargs"]["eps"] = sweep_params["optimizer_eps"]

        log.info(f"run config: {pprint.pformat(RUN_CONFIG)}")

        env_kwargs = {
            "jps_instance": jsp_instance,
            "scaling_divisor": jsp_instance_details["lower_bound"],
            **RUN_CONFIG["env_kwargs"]
        }

        log.info(f"setting up vectorised environment")

        def mask_fn(env):
            return env.valid_action_mask()

        venv = make_vec_env_without_monitor(
            env_id=RUN_CONFIG["env_name"],
            env_kwargs=env_kwargs,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn},
            n_envs=RUN_CONFIG["n_envs"]
        )

        venv = CuriosityInfoWrapper(venv=venv)

        venv = VecMonitor(venv=venv)

        log.info(f"setting up mask ppo model")

        model = sb3_contrib.MaskablePPO(
            RUN_CONFIG["policy_type"],
            env=venv,
            verbose=1,
            tensorboard_log=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
            **RUN_CONFIG["model_hyper_parameters"]
        )

        wb_cb = WandbCallback(
            gradient_save_freq=100,
            model_save_path=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
            verbose=1,
        )
        logger_cb = JssLoggerCallback(wandb_ref=wb)

        log.info(f"training the agent")
        model.learn(
            total_timesteps=RUN_CONFIG["total_timesteps"],
            callback=[
                wb_cb,
                logger_cb
            ]
        )

        log.info("evaluating model performance")
        n_eval_episodes = sweep_params["n_eval_episodes"]
        makespans = []

        eval_env_kwargs = env_kwargs.copy()
        eval_env_kwargs["perform_left_shift_if_possible"] = True

        venv = make_vec_env_without_monitor(
            env_id=RUN_CONFIG["env_name"],
            env_kwargs=eval_env_kwargs,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn},
            n_envs=RUN_CONFIG["n_envs"]
        )

        venv = VecMonitor(venv=venv)

        model.set_env(venv)

        for _ in track(range(n_eval_episodes), description="evaluating model performance ..."):
            done = False
            obs = venv.reset()
            while not done:
                masks = np.array([env.action_masks() for env in model.env.envs])
                action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
                obs, rewards, dones, info = venv.step(action)
                done = np.all(dones == True)
                if done:
                    for sub_env_info in info:
                        makespans.append(sub_env_info["makespan"])

        from statistics import mean
        mean_return = mean(makespans)

        log.info(f"mean evaluation makespan: {mean_return:.2f}")
        wb.log({"mean_makespan": mean_return})

        obs = venv.reset()
        venv.close()
        del venv


if __name__ == '__main__':
    # sweep_id = wb.sweep(SWEEP_CONFIG, project=PROJECT)
    sweep_id = "cny03n9f"
    log.info(f"use this 'sweep_id': '{sweep_id}'")
    wb.agent(sweep_id, function=perform_run, count=200, project=PROJECT)
