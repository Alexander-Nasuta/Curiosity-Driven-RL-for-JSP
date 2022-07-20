from typing import Dict

import sb3_contrib
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

import jss_utils.jsp_env_utils as env_utils

from jss_utils.jss_logger import log
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env


def venv_basic_ppo_example(env_kwargs: Dict, total_timesteps=1_000, n_envs: int=4):
    log.info("setting up vectorised environment")

    def mask_fn(env):
        return env.valid_action_mask()

    venv = make_vec_env(
        env_id=DisjunctiveGraphJssEnv,
        env_kwargs=env_kwargs,

        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},

        n_envs=n_envs)

    model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env=venv,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)


if __name__ == '__main__':
    jsp, lb = env_utils.get_benchmark_instance_and_lower_bound("ft06")
    venv_basic_ppo_example(env_kwargs={
        "jps_instance": jsp,
        "scaling_divisor": lb,
        "perform_left_shift_if_possible": True,
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True
    })


