import time

import sb3_contrib
import gym

import numpy as np

import pathlib as pl

from matplotlib import pyplot as plt
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env

import jss_utils.PATHS as PATHS
import jss_utils.jsp_instance_parser as parser

from jss_utils.jss_logger import log
from jss_graph_env.disjunctive_graph_jss_visualizer import DisjunctiveGraphJspVisualizer

gym.envs.register(
    id='GraphJsp-v0',
    entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
    kwargs={},
)


def solve_jsp(jsp_instance: np.ndarray, path: pl.Path, env_kwargs: dict, n_envs: int, plot_results: bool = True):
    def mask_fn(env):
        return env.valid_action_mask()

    env_kwargs = {
        "jps_instance": jsp_instance,
        **env_kwargs,
        "scale_reward": False  # return makespan as in the end
    }

    venv = make_vec_env(
        env_id='GraphJsp-v0',
        env_kwargs=env_kwargs,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        n_envs=n_envs
    )

    model = sb3_contrib.MaskablePPO.load(env=venv, path=path)

    start = time.perf_counter()

    done = False
    obs = venv.reset()
    while not done:
        masks = np.array([env.action_masks() for env in model.env.envs])
        action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
        obs, rewards, dones, info = venv.step(action)
        done = np.all(dones == True)

        if done:
            end = time.perf_counter()
            solving_duration = end - start

            # index_of_best_solution
            i, makespan = max(enumerate(rewards), key=lambda data: data[1])
            makespan = -makespan  # shift sign
            info = info[i]
            df = info["gantt_df"]
            if plot_results:
                # generate colors for visualizer
                _, _, machines_count = jsp_instance.shape
                c_map = plt.cm.get_cmap("rainbow")  # select the desired cmap
                arr = np.linspace(0, 1, machines_count)  # create a list with numbers from 0 to 1 with n items
                machine_colors = {m_id: c_map(val) for m_id, val in enumerate(arr)}
                colors = {f"Machine {m_id}": (r, g, b) for m_id, (r, g, b, a) in machine_colors.items()}

                visualizer = DisjunctiveGraphJspVisualizer(dpi=80)
                visualizer.gantt_chart_console(df, colors)

            log.info(f"makespan: {makespan}")
            log.info(f"solving duration: {solving_duration:2f} sec")
            # return makespan, df, info[i]


if __name__ == '__main__':
    run_name = "divine-totem-165"
    run_id = "1nl7ebtr"
    model_path = PATHS.WANDB_PATH.joinpath(f"{run_name}_{run_id}").joinpath("model")

    jsp = parser.get_instance_by_name('ft06')

    env__kwargs = {
        'default_visualisations': ['gantt_window'],
        'dtype': 'float32',
        'flat_observation_space': True,
        'normalize_observation_space': True,
        'perform_left_shift_if_possible': True,
        'scale_reward': True
    }
    solve_jsp(
        jsp_instance=jsp,
        path=model_path,
        env_kwargs=env__kwargs,
        n_envs=8
    )
