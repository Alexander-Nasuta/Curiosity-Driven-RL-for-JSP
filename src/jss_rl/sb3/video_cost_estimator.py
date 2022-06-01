import time
import datetime

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

N_ENVS = 4
VISUALISATIONS = [
    "gantt_window",
    "graph_window",  # very expensive
]

if __name__ == '__main__':
    jsp, lb = env_utils.get_benchmark_instance_and_lower_bound("ft06")

    env_kwargs = {
        # placeholder for action and observation space shape
        "jps_instance": jsp,
        "scaling_divisor": lb,
        "perform_left_shift_if_possible": True,
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True,
        "default_visualisations": VISUALISATIONS
    }

    log.info("setting up vectorised environment")


    def mask_fn(env):
        return env.valid_action_mask()


    venv = make_vec_env(
        env_id=DisjunctiveGraphJssEnv,
        env_kwargs=env_kwargs,

        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},

        n_envs=N_ENVS)

    model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env=venv,
        verbose=1,
    )

    total_timesteps = 20_000

    # model.learn(total_timesteps=total_timesteps)

    log.info("setting up video recorder")
    video_len = 10

    venv = VecVideoRecorder(
        venv,
        PATHS.SB3_EXAMPLES_VIDEO,
        record_video_trigger=lambda x: x == 0,
        video_length=video_len,
        name_prefix=f"video_cost_estimator")

    obs = venv.reset()
    start = time.perf_counter()
    for _ in track(range(video_len), description="recording frames ..."):
        masks = np.array([env.action_masks() for env in model.env.envs])
        action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
        obs, _, _, _ = venv.step(action)

    end = time.perf_counter()

    recording_duration = end - start
    log.info(f"recording duration: {recording_duration:2f} sec for {video_len} frames using {N_ENVS} environments")
    for i in range(5, 101, 5):
        dur = datetime.timedelta(seconds=int(i * recording_duration))
        log.info(f"cost for {i * video_len:>4} frames: {dur}")

    # somehow VecVideoRecorder crashes at the end of the script (when __del__() in VecVideoRecorder is called)
    # for some reason there are no issues when deleting env manually
    del venv
    log.info("done.")