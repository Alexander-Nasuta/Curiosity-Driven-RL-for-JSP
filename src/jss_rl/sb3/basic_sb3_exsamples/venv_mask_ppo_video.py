import sys

import numpy as np
import sb3_contrib

import jss_utils.PATHS as PATHS
import jss_utils.jsp_env_utils as env_utils

from jss_utils.jss_logger import log
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

from rich.progress import track
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env

if __name__ == '__main__':
    jsp, lb = env_utils.get_benchmark_instance_and_lower_bound("ft06")

    env_kwargs = {
        # placeholder for action and observation space shape
        "jps_instance": jsp,
        "scaling_divisor": lb,
        "perform_left_shift_if_possible": True,
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True
    }

    log.info("setting up vectorised environment")


    def mask_fn(env):
        return env.valid_action_mask()


    venv = make_vec_env(
        env_id=DisjunctiveGraphJssEnv,
        env_kwargs=env_kwargs,

        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},

        n_envs=4)

    model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env=venv,
        verbose=1,
    )

    total_timesteps = 20_000

    model.learn(total_timesteps=total_timesteps)

    log.info("setting up video recorder")
    episode_len = venv.envs[0].total_tasks_without_dummies

    venv = VecVideoRecorder(
        venv,
        PATHS.SB3_EXAMPLES_VIDEO,
        record_video_trigger=lambda x: x == (total_timesteps-episode_len),
        video_length=episode_len,
        name_prefix=f"venv_mask_ppo_video")

    obs = venv.reset()
    for _ in track(range(episode_len), description="recording frames ..."):
        masks = np.array([env.action_masks() for env in model.env.envs])
        action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
        obs, _, _, _ = venv.step(action)

    # Save the video
    log.info("saving video...")
    venv.close()

    # somehow VecVideoRecorder crashes at the end of the script (when __del__() in VecVideoRecorder is called)
    # for some reason there are no issues when deleting env manually
    del venv
    log.info("done.")
