import time

import numpy as np

import jss_utils.jsp_env_utils as env_utils
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

from jss_utils.jss_logger import log


def trivial_schedule(jsp_instance: np.ndarray, lower_bound: int) -> None:
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

    done = False
    score = 0

    iteration_count = 0
    start = time.perf_counter()
    for i in range(env.total_tasks_without_dummies):
        n_state, reward, done, info = env.step(i)
        score += reward
        iteration_count += 1

    end = time.perf_counter()

    env.render(show=["gantt_console", "graph_console"])

    log.info(f"score: {score}")
    total_duration = end - start
    log.info(f"total duration: {total_duration:2f} sec")
    log.info(f"average iteration duration: {total_duration / iteration_count:4f} sec")


if __name__ == '__main__':
    jsp, lb = env_utils.get_benchmark_instance_and_lower_bound(name="abz5")
    trivial_schedule(jsp_instance=jsp, lower_bound=lb)
