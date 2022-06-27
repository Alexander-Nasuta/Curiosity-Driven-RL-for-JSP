import datetime

import gym
import pprint
import time

import numpy as np
import sb3_contrib

import wandb as wb
import torch as th
from rich.progress import track

import jss_utils.PATHS as PATHS
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from wandb.integration.sb3 import WandbCallback
from jss_rl.sb3.util.callbacks.episode_end_moving_average_rollout_end_logger_callback import \
    EpisodeEndMovingAverageRolloutEndLoggerCallback

from jss_utils.jss_logger import log

PROJECT = "JSP-test"
BENCHMARK_INSTANCE_NAME = "ft06"

wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))

jsp_instance = parser.get_instance_by_name(BENCHMARK_INSTANCE_NAME)
jsp_instance_details = details.get_jps_instance_details(BENCHMARK_INSTANCE_NAME)

_, n_jobs, n_machines = jsp_instance.shape

gym.envs.register(
    id='GraphJsp-v0',
    entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
    kwargs={},
)

config = {
    "total_timesteps": 100_000,
    "n_envs": 8,  # multiprocessing.cpu_count()-1

    "instance_name": BENCHMARK_INSTANCE_NAME,
    "instance_details": jsp_instance_details,
    "n_jobs": n_jobs,
    "n_machines": n_machines,

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
        "dtype": "float32",
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

if __name__ == '__main__':
    log.info(f"config: {pprint.pformat(config)}")

    start = time.perf_counter()

    run = wb.init(
        project=PROJECT,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        dir=f"{PATHS.WANDB_PATH}/",
    )

    log.info(f"run name: {run.name}, run id: {run.id}")

    env_kwargs = {
        "jps_instance": jsp_instance,
        "scaling_divisor": jsp_instance_details["lower_bound"],
        **config["env_kwargs"]
    }

    log.info(f"setting up vectorised environment")


    def mask_fn(env):
        return env.valid_action_mask()


    venv = make_vec_env(
        env_id=config["env_name"],
        env_kwargs=env_kwargs,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        n_envs=config["n_envs"]
    )

    log.info(f"setting up mask ppo model")

    model = sb3_contrib.MaskablePPO(
        config["policy_type"],
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

    logger_cb = EpisodeEndMovingAverageRolloutEndLoggerCallback(
        **config["EpisodeEndMovingAverageRolloutEndLoggerCallback_kwargs"],
    )

    log.info(f"training the agent")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wb_cb, logger_cb]
    )

    # somehow the mask ppo does not work trigger properly. the step appears to count only to the batch size and then
    # start again at step 0
    # therefore here is a workaround
    log.info(f"setting up video recorder")

    _, n_jobs, n_machines = jsp_instance.shape
    episode_len = n_jobs * n_machines

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

    video_file = next(video_folder.glob('*.mp4'))
    wb_video = wb.Video(data_or_path=str(video_file))
    wb.log({"video": wb_video})

    del venv
    run.finish()

    end = time.perf_counter()
    solving_duration = end - start

    for i in range(25, 2_001, 25):
        dur = datetime.timedelta(seconds=int(i * solving_duration))
        log.info(f"cost for {i} runs with {config['total_timesteps']} timesteps and {config['n_envs']} envs: {dur}")
