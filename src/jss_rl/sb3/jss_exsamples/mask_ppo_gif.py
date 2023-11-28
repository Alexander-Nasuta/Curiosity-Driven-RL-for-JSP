import gym
import imageio
import sb3_contrib

import numpy as np
import stable_baselines3 as sb3

import jss_utils.PATHS as PATHS
import jss_utils.jsp_env_utils as env_utils
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

from jss_utils.jss_logger import log

from rich.progress import track
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


def train_and_create_gif(jsp_instance: np.ndarray, lower_bound: int,
                         total_timesteps: int = 2_000, filename: str = "mask_ppo_gif") -> None:
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

    images = []
    obs = model.env.reset()
    img = model.env.render(mode='rgb_array')
    images.append(img)
    for _ in track(range(model.env.envs[0].total_tasks_without_dummies), description="creating gif..."):
        action, _ = model.predict(
            obs,
            action_masks=model.env.envs[0].action_masks(),
            deterministic=False
        )
        obs, _, done, _ = model.env.step(action)
        img = model.env.render(mode='rgb_array')
        images.append(img)

    imageio.mimsave(
        PATHS.SB3_EXAMPLES_GIF.joinpath(f'{filename}.gif'),
        [np.array(img) for i, img in enumerate(images)],
        fps=10
    )

    log.info("done")


if __name__ == '__main__':
    jsp, lb = env_utils.get_benchmark_instance_and_lower_bound(name="ft06")
    train_and_create_gif(jsp_instance=jsp, lower_bound=lb)
