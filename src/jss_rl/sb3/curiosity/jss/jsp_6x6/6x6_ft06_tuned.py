import pprint
from statistics import mean
from typing import List

import gym
import sb3_contrib

import numpy as np
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


def main(num_runs: int = 10):
    PROJECT = "6x6_fto6_tuned_dev"
    BENCHMARK_INSTANCE_NAME = "ft06"

    jsp_instance, jsp_instance_details = env_utils.get_benchmark_instance_and_details(name="ft06")

    _, n_jobs, n_machines = jsp_instance.shape

    wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))

    RUN_CONFIG = {
        "total_timesteps": 100_000,
        "n_envs": 8,  # multiprocessing.cpu_count()-1

        "instance_name": BENCHMARK_INSTANCE_NAME,
        "instance_details": jsp_instance_details,

        "policy_type": MaskableActorCriticPolicy,
        "model_hyper_parameters": {
            "gamma": 0.99999,  # discount factor,
            "gae_lambda": 0.95,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "clip_range": 0.541,
            "clip_range_vf": None,
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

    # plain Mask PPO
    log.info(f"['Mask PPO'] RUN_CONFIG: {pprint.pformat(RUN_CONFIG)}")
    for _ in track(range(num_runs), description="running experiments with plain Mask PPO"):

        run = wb.init(
            project=PROJECT,
            group="Mask PPO",
            config=RUN_CONFIG,
            sync_tensorboard=True,
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
            dir=f"{PATHS.WANDB_PATH}/",
        )

        log.info(f"run name: {run.name}, run id: {run.id}")

        wb.log({
            "tasks_to_machines_mapping": wb.Table(
                data=jsp_instance[0],
                columns=[f"task #{i}" for i in range(n_machines)]
            )
        })

        wb.log({
            "tasks_to_duration_mapping": wb.Table(
                data=jsp_instance[1],
                columns=[f"task #{i}" for i in range(n_machines)]
            )
        })

        env_kwargs = {
            "jps_instance": jsp_instance,
            "scaling_divisor": jsp_instance_details["lower_bound"],
            **RUN_CONFIG["env_kwargs"]
        }

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

        logger_cb = JssLoggerCallback()

        model.learn(total_timesteps=RUN_CONFIG["total_timesteps"], callback=[wb_cb, logger_cb])

        log.info(f"setting up video recorder")

        episode_len = n_jobs * n_machines

        video_folder = PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}")

        venv = VecVideoRecorder(
            venv=venv,
            video_folder=video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=episode_len,
            name_prefix=f"{run.name}_{run.id}")

        obs = venv.reset()
        infos = None

        log.info("recording frames ...")
        for _ in range(episode_len):
            masks = np.array([env.action_masks() for env in model.env.envs])
            action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
            obs, _, _, infos = venv.step(action)

        for i in range(venv.num_envs):
            wb.log({
                f'gantt_df_of_env_{i}': infos[i]["gantt_df"]
            })

        # Save the video
        log.info("logging video to wandb video...")
        venv.close()

        # video is saved automatically, if monitor_gym=True (see wb.init above)
        # video_file = next(video_folder.glob('*.mp4'))
        # wb_video = wb.Video(data_or_path=str(video_file))
        # wb.log({"video": wb_video})

        run.finish()
        del venv

    RUN_CONFIG = RUN_CONFIG.copy()
    RUN_CONFIG["IntrinsicCuriosityModuleWrapper"] = {
        "beta": 0.25,
        "eta": 0.01,
        "lr": 0.0003,
        "device": 'cpu',
        "feature_dim": 288,
        "feature_net_hiddens": [256, 256],
        "feature_net_activation": "relu",
        "inverse_feature_net_hiddens": [256, 256],
        "inverse_feature_net_activation": "relu",
        "forward_fcnet_net_hiddens": [256, 256],
        "forward_fcnet_net_activation": "relu",

        # "maximum_sample_size": 16,

        "clear_memory_on_end_of_episode": True,
        "postprocess_on_end_of_episode": True,

        "clear_memory_every_n_steps": None,
        "postprocess_every_n_steps": None
    }

    log.info(f"['Mask PPo + ICM'] RUN_CONFIG: {pprint.pformat(RUN_CONFIG)}")

    for _ in track(range(num_runs), description="running experiments with plain 'Mask PPO + ICM' "):

        run = wb.init(
            project=PROJECT,
            group="Mask PPO + ICM",
            config=RUN_CONFIG,
            sync_tensorboard=True,
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
            dir=f"{PATHS.WANDB_PATH}/",
        )

        log.info(f"run name: {run.name}, run id: {run.id}")

        wb.log({
            "tasks_to_machines_mapping": wb.Table(
                data=jsp_instance[0],
                columns=[f"task #{i}" for i in range(n_machines)]
            )
        })

        wb.log({
            "tasks_to_duration_mapping": wb.Table(
                data=jsp_instance[1],
                columns=[f"task #{i}" for i in range(n_machines)]
            )
        })

        env_kwargs = {
            "jps_instance": jsp_instance,
            "scaling_divisor": jsp_instance_details["lower_bound"],
            **RUN_CONFIG["env_kwargs"]
        }

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

        venv = IntrinsicCuriosityModuleWrapper(
            venv=venv,
            **RUN_CONFIG["IntrinsicCuriosityModuleWrapper"]
        )

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

        logger_cb = JssLoggerCallback()

        model.learn(total_timesteps=RUN_CONFIG["total_timesteps"], callback=[wb_cb, logger_cb])

        log.info(f"setting up video recorder")

        episode_len = n_jobs * n_machines

        video_folder = PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}")

        venv = VecVideoRecorder(
            venv=venv,
            video_folder=video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=episode_len,
            name_prefix=f"{run.name}_{run.id}")

        obs = venv.reset()
        infos = None

        log.info("recording frames ...")
        for _ in range(episode_len):
            masks = np.array([env.action_masks() for env in model.env.envs])
            action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
            obs, _, _, infos = venv.step(action)

        for i in range(venv.num_envs):
            wb.log({
                f'gantt_df_of_env_{i}': infos[i]["gantt_df"]
            })

        # Save the video
        log.info("logging video to wandb video...")
        venv.close()

        # video is saved automatically, if monitor_gym=True (see wb.init above)
        # video_file = next(video_folder.glob('*.mp4'))
        # wb_video = wb.Video(data_or_path=str(video_file))
        # wb.log({"video": wb_video})

        run.finish()
        del venv


if __name__ == '__main__':
    main(num_runs=1)
