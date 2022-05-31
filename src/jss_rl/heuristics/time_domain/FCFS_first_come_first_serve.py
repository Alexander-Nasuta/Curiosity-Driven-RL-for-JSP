import time

import numpy as np
import networkx as nx
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

from jss_utils.jss_logger import log
import jss_utils.jsp_instance_parser as parser


def solve_jsp(jsp_instance: np.ndarray, plot_results: bool = True):

    env = DisjunctiveGraphJssEnv(
        jps_instance=jsp_instance,
        perform_left_shift_if_possible=False,
        scaling_divisor=None,
        scale_reward=False,
        normalize_observation_space=True,
        flat_observation_space=True
    )

    done = False

    start = time.perf_counter()
    while not done:
        possible_actions = env.valid_action_mask()

        valid_tasks_path = [
            -nx.shortest_path_length(env.G, env.src_task, i, method='bellman-ford', weight='nweight')
            for i, bol in enumerate(possible_actions, start=1)
            if bol]
        valid_tasks = [i for i, bol in enumerate(possible_actions) if bol]

        action = valid_tasks[np.argmin(valid_tasks_path)]
        n_state, reward, done, info = env.step(action)

    end = time.perf_counter()
    solving_duration = end - start

    if plot_results:
        env.render(show=["gantt_console"])
        log.info(f"solving duration: {solving_duration:2f} sec")

    df = env.network_as_dataframe()
    makespan = info["makespan"]
    info["FCFS_solving_duration"] = solving_duration
    info["FCFS_makespan"] = solving_duration

    return makespan, df, info


if __name__ == '__main__':
    import jss_utils.jsp_or_tools_solver as or_solver

    jsp = parser.get_instance_by_name("ta01")

    optimal_makespan, *_ = or_solver.solve_jsp(jsp_instance=jsp, plot_results=False)
    makespan, *_ = solve_jsp(jsp_instance=jsp)

    log.info(f"makespan to optimal makespan ratio: {makespan / optimal_makespan:.2f}")
