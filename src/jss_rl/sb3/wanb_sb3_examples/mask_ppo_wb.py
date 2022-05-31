import gym
import sb3_contrib

import wandb as wb
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

import jss_utils.PATHS as PATHS
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details

from jss_utils.jss_logger import log

PROJECT = "JSP-test"
BENCHMARK_INSTANCE_NAME = "ft06"

wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))

jsp_instance = parser.get_instance_by_name(BENCHMARK_INSTANCE_NAME)
jsp_instance_details = details.get_jps_instance_details(BENCHMARK_INSTANCE_NAME)

gym.envs.register(
        id='GraphJsp-v0',
        entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
        kwargs={},
)

config = {
    "instance_name": BENCHMARK_INSTANCE_NAME,
    "instance_details": jsp_instance_details,

    "policy_type": MaskableActorCriticPolicy,

    "total_timesteps": 10_000,
    "n_envs": 4, #multiprocessing.cpu_count()-1

    "env_name": "GraphJsp-v0",
    "env_kwargs": {
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True,
        "perform_left_shift_if_possible": True,
    }
}


if __name__ == '__main__':
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

    log.info(f"setting up video recorder")

    _, n_jobs, n_machines = jsp_instance.shape
    episode_len = n_jobs * n_machines

    venv = VecVideoRecorder(
        venv=venv,
        video_folder=PATHS.WANDB_PATH,
        record_video_trigger=lambda x: x == (config["total_timesteps"] - episode_len),
        video_length=episode_len,
        name_prefix=f"{run.name}_{run.id}")

    log.info(f"setting up mask ppo model")

    model = sb3_contrib.MaskablePPO(
        config["policy_type"],
        env=venv,
        verbose=1,
        tensorboard_log=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}")
    )

    wb_cb = WandbCallback(
        gradient_save_freq=100,
        model_save_path=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
        verbose=1,
    )

    log.info(f"training the agent")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=wb_cb
    )

    run.finish()