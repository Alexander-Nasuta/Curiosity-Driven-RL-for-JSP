import gym
import sb3_contrib

import numpy as np
import stable_baselines3 as sb3

import jss_utils.jsp_env_utils as env_utils

from jss_utils.jss_logger import log

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


if __name__ == '__main__':
    env = env_utils.get_pre_configured_example_env(
        name="ft06",
        perform_left_shift_if_possible=True,
        scale_reward=True,
        normalize_observation_space=True,
        flat_observation_space=True
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

    # Train the agent for 20 000 steps
    log.info("training the model")
    model.learn(total_timesteps=20_000)

    # NOTE: evaliation seems not to use the action mask
    # log.info("evaluating the model")
    # mean, std = sb3.common.evaluation.evaluate_policy(model, env, n_eval_episodes=10, deterministic=False)
    # log.info(f"Model mean reward: {mean:.2f}, std: {std:.2f}")