import pprint
import gym
import numpy as np
import sb3_contrib

import torch as th
import wandb as wb
from rich.progress import track
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

import jss_utils.PATHS as PATHS
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details
from jss_rl.sb3.util.callbacks.episode_end_moving_average_rollout_end_logger_callback import \
    EpisodeEndMovingAverageRolloutEndLoggerCallback

from jss_utils.jss_logger import log

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

PROJECT = "mask_ppo_test1"
BENCHMARK_INSTANCE_NAME = "ft06"

wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))

jsp_instance = parser.get_instance_by_name(BENCHMARK_INSTANCE_NAME)
jsp_instance_details = details.get_jps_instance_details(BENCHMARK_INSTANCE_NAME)

gym.envs.register(
    id='GraphJsp-v0',
    entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
    kwargs={},
)

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'mean_return',
        'goal': 'maximize'
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
            "values": [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
        },
        "batch_size": {
            'values': [16, 32, 64, 128, 256, 512, 1024, 2048]
        },
        "clip_range": {
            'values': [0.02, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        },
        "clip_range_vf": {
            'values': [26]
        },
        "ent_coef": {
            "values": [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5, 1e-6, 1e-7, 1e-8]
        },
        "normalize_advantage": {
            'values': [True, False]
        },
        # "target_kl": 0.05047, # for early stopping
        "net_arch_n_layers": {
            'values': [2, 3, 4]
        },
        "net_arch_n_size": {
            'values': [8, 16, 32, 64, 128, 256]
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
        "normalize_observation_space": {
            'values': [True, False]
        },
        "flat_observation_space": {
            'values': [True, False]
        },
        "perform_left_shift_if_possible": {
            'values': [True, False]
        },
        "dtype": {
            'values': ["float32"]
        },

        # eval params
        "n_eval_episodes": {
            'value': 10
        }
    }
}

RUN_CONFIG = {
    "total_timesteps": 100_000,
    "n_envs": 8,  # multiprocessing.cpu_count()-1

    "instance_name": BENCHMARK_INSTANCE_NAME,
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

    "EpisodeEndMovingAverageRolloutEndLoggerCallback_kwargs": {
        "fields": [
            "extrinsic_reward",
            "makespan",
            "scaling_divisor"
        ],
        "capacity": 100,
    },

}


def perform_run():
    with wb.init(
            sync_tensorboard=True,
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
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

        venv = make_vec_env(
            env_id=RUN_CONFIG["env_name"],
            env_kwargs=env_kwargs,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn},
            n_envs=RUN_CONFIG["n_envs"]
        )

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

        logger_cb = EpisodeEndMovingAverageRolloutEndLoggerCallback(
            **RUN_CONFIG["EpisodeEndMovingAverageRolloutEndLoggerCallback_kwargs"],
        )

        log.info(f"training the agent")
        model.learn(
            total_timesteps=RUN_CONFIG["total_timesteps"],
            callback=[wb_cb, logger_cb]
        )

        _, n_jobs, n_machines = jsp_instance.shape
        episode_len = n_jobs * n_machines

        log.info("evaluating model performance")
        n_eval_episodes = sweep_params["n_eval_episodes"]
        eval_rewards = []
        for _ in track(range(n_eval_episodes), description="evaluating model performance ..."):
            done = False
            obs = venv.reset()
            while not done:
                masks = np.array([env.action_masks() for env in model.env.envs])
                action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
                obs, rewards, dones, info = venv.step(action)
                done = np.all(dones == True)
                if done:
                    for r in rewards:
                        eval_rewards.append(r)

        from statistics import mean
        mean_return = mean(eval_rewards)

        log.info(f"mean evaluation return: {mean_return:.2f}")
        wb.log({"mean_return": mean_return})

        # somehow the mask ppo does not work trigger properly. the step appears to count only to the batch size and then
        # start again at step 0
        # therefore here is a workaround
        log.info(f"setting up video recorder")

        video_folder = PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}")

        venv = VecVideoRecorder(
            venv=venv,
            video_folder=video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=episode_len,
            name_prefix=f"{run.name}_{run.id}")

        obs = venv.reset()
        for _ in track(range(episode_len), description="recording frames ..."):
            masks = np.array([env.action_masks() for env in model.env.envs])
            action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
            obs, _, _, _ = venv.step(action)

        # Save the video
        log.info("saving video...")
        venv.close()


if __name__ == '__main__':
    
    sweep_id = wb.sweep(sweep_config, project=PROJECT)
    # sweep_id = "5r0oucpj"
    log.info(f"use this 'sweep_id': '{sweep_id}'")
    wb.agent(sweep_id, function=perform_run, count=10, project=PROJECT)
