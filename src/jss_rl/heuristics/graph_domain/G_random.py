import time
import random

import numpy as np

import jss_utils.jsp_instance_parser as parser


from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv
from jss_utils.jss_logger import log


def solve_jsp(jsp_instance: np.ndarray, plot_results: bool = True):
    env = DisjunctiveGraphJssEnv(
        jps_instance=jsp_instance,
        perform_left_shift_if_possible=False,
        scaling_divisor=None,
        scale_reward=False,
        normalize_observation_space=True,
        flat_observation_space=True,
    )

    done = False

    start = time.perf_counter()
    while not done:
        possible_actions = env.valid_action_mask()
        valid_tasks = [i for i, bol in enumerate(possible_actions) if bol]

        action = random.choice(valid_tasks)
        n_state, reward, done, info = env.step(action)

    end = time.perf_counter()
    solving_duration = end - start

    if plot_results:
        env.render(show=["gantt_console"])
        log.info(f"solving duration: {solving_duration:2f} sec")

    makespan = info["makespan"]
    info["solving_duration"] = solving_duration
    info["G_random_solving_duration"] = solving_duration
    info["G_random_makespan"] = solving_duration

    return makespan, info


if __name__ == '__main__':
    import jss_utils.jsp_or_tools_solver as or_solver

    jsp = parser.get_instance_by_name("ft06")

    optimal_makespan, *_ = or_solver.solve_jsp(jsp_instance=jsp, plot_results=False)
    makespan, *_ = solve_jsp(jsp_instance=jsp)

    log.info(f"makespan to optimal makespan ratio: {makespan / optimal_makespan:.2f}")

