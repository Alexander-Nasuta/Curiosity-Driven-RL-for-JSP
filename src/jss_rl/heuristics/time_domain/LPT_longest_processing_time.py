import sys
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
    t = 0
    info = None

    start = time.perf_counter()

    job_processing_times = np.sum(jsp_instance[1], axis=1)

    while not done:
        possible_actions = env.valid_action_mask()
        valid_tasks = [i for i, bol in enumerate(possible_actions, start=1) if bol]  # node/task to action shift
        machine_available = {
            m_id: not len(m_route) or t >= env.G.nodes[m_route[-1]]["finish_time"]
            for m_id, m_route in env.machine_routes.items()
        }
        candidates = [
            (i, -nx.shortest_path_length(env.G, env.src_task, i, method='bellman-ford', weight='nweight'))
            for i in valid_tasks
            if machine_available[env.G.nodes[i]["machine"]]
        ]
        candidates = [i for i, release_time in candidates if t >= release_time]

        if candidates:
            candidates = [
                (i, job_processing_times[(i-1) // env.n_machines])
                for i in candidates
            ]
            action, _ = max(candidates, key=lambda data: data[1])
            action = action - 1  # node/task to action shift
            n_state, reward, done, info = env.step(action=action)
            continue

        t += 1

        if t > env.horizon:
            raise RuntimeError("time step exceeded horizont")

    end = time.perf_counter()
    solving_duration = end - start

    if plot_results:
        env.render(show=["gantt_console"])
        log.info(f"solving duration: {solving_duration:2f} sec")

    makespan = info["makespan"]
    info["solving_duration"] = solving_duration
    info["LPT_solving_duration"] = solving_duration
    info["LPT_makespan"] = solving_duration

    return makespan, info


if __name__ == '__main__':
    import jss_utils.jsp_or_tools_solver as or_solver

    jsp = parser.get_instance_by_name("ft06")

    optimal_makespan, *_ = or_solver.solve_jsp(jsp_instance=jsp, plot_results=False)
    makespan, *_ = solve_jsp(jsp_instance=jsp)

    log.info(f"makespan to optimal makespan ratio: {makespan / optimal_makespan:.2f}")
