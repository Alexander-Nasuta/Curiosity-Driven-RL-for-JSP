import gym
import sb3_contrib

import numpy as np
import stable_baselines3 as sb3

import jss_utils.jsp_env_utils as env_utils
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

from jss_utils.jss_logger import log

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


def jss_env_mask_ppo_example(jsp_instance: np.ndarray, lower_bound: int,
                             total_timesteps: int = 20_000) -> None:
    env = DisjunctiveGraphJssEnv(
        jps_instance=jsp_instance,
        perform_left_shift_if_possible=True,
        scaling_divisor=lower_bound,
        scale_reward=True,
        normalize_observation_space=True,
        flat_observation_space=True,
        action_mode='task',  # alternative 'job'
        dtype='float32'
    )

    env = sb3.common.monitor.Monitor(env)

    def mask_fn(env: gym.Env) -> np.ndarray:
        return env.valid_action_mask()

    env = ActionMasker(env, mask_fn)

    model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1
    )

    # Train the agent
    log.info("training the model")
    model.learn(total_timesteps=total_timesteps)

    # NOTE: evaliation seems not to use the action mask
    # log.info("evaluating the model")
    # mean, std = sb3.common.evaluation.evaluate_policy(model, env, n_eval_episodes=10, deterministic=False)
    # log.info(f"Model mean reward: {mean:.2f}, std: {std:.2f}")


if __name__ == '__main__':
    jsp, lb = env_utils.get_benchmark_instance_and_lower_bound(name="ft06")
    jss_env_mask_ppo_example(jsp_instance=jsp, lower_bound=lb)