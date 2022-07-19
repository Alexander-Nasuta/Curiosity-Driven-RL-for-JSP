import time

import numpy as np

import jss_utils.jsp_instance_parser as parser

from collections import Counter

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
        action_mode='job'
    )

    done = False

    start = time.perf_counter()
    while not done:
        tasks = [task - 1 for _, m_route in env.machine_routes.items() for task in m_route]  # action task index shift
        tasks = [task // env.n_machines for task in tasks]  # map task to job
        counter = {i: 0 for i in range(env.n_machines)}  # fallback values
        counter = {**counter, **Counter(tasks)}  # count occurrences
        job_id, _ = min(counter.items(), key=lambda data: data[1])  # pick the job with the least occurrences
        n_state, reward, done, info = env.step(job_id)

    end = time.perf_counter()
    solving_duration = end - start

    if plot_results:
        env.render(show=["gantt_console"])
        log.info(f"solving duration: {solving_duration:2f} sec")

    makespan = info["makespan"]
    info["solving_duration"] = solving_duration
    info["GLNRT_solving_duration"] = solving_duration
    info["GLNRT_makespan"] = solving_duration

    return makespan, info


if __name__ == '__main__':
    import jss_utils.jsp_or_tools_solver as or_solver

    jsp = parser.get_instance_by_name("ft06")

    optimal_makespan, *_ = or_solver.solve_jsp(jsp_instance=jsp, plot_results=False)
    makespan, *_ = solve_jsp(jsp_instance=jsp)

    log.info(f"makespan to optimal makespan ratio: {makespan / optimal_makespan:.2f}")
