import pprint

import gym
import sb3_contrib

import numpy as np
import torch as th
import wandb as wb

import jss_utils.PATHS as PATHS
import jss_utils.jsp_env_utils as env_utils

from rich.progress import track

from wandb.integration.sb3 import WandbCallback

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder

from jss_rl.sb3.curiosity_modules.curiosity_info_wrapper import CuriosityInfoWrapper
from jss_rl.sb3.curiosity.jss.jss_logger_callback import JssLoggerCallback
from jss_rl.sb3.util.callbacks.dynamic_custom_instance_loader_callback import DynamicCustomInstanceLoaderCallback
from jss_rl.sb3.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_utils.jss_logger import log

wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))

gym.envs.register(
    id='GraphJsp-v0',
    entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
    kwargs={},
)


def main(num_runs: int = 10):
    PROJECT = "6x6_dynamic_instance_dev"

    RUN_CONFIG = {
        "load_instance_interval": 5,

        "total_timesteps": 5 * 100_000,
        "n_envs": 8,  # multiprocessing.cpu_count()-1


        "policy_type": MaskableActorCriticPolicy,
        "model_hyper_parameters": {
            "gamma": 0.99999,  # discount factor,
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

        # will be overwritten by callback
        jsp_instance, jsp_instance_details = env_utils.get_benchmark_instance_and_details(name="ft06")
        _, n_jobs, n_machines = jsp_instance.shape

        env_kwargs = {
            "jps_instance": jsp_instance, # will be overwritten by callback
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

        instance_loader_cb = DynamicCustomInstanceLoaderCallback(
            load_instance_every_n_rollouts=RUN_CONFIG["load_instance_interval"],
            wandb_ref=wb
        )

        logger_cb = JssLoggerCallback(wandb_ref=wb)

        model.learn(
            total_timesteps=RUN_CONFIG["total_timesteps"],
            callback=[wb_cb, logger_cb, instance_loader_cb]
        )

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
