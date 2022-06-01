import time

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
        action_mode='job'
    )

    done = False

    start = time.perf_counter()
    while not done:
        task_mask = env.valid_action_mask(action_mode='task')
        masks_per_job = np.array_split(task_mask, env.n_jobs)
        actions = np.array([np.inf] * env.n_jobs)
        for job_index, mask in enumerate(masks_per_job):
            if True not in mask:
                continue
            ture_index = np.argmax(mask)
            start_at = 1 + job_index * env.n_machines + ture_index
            stop_at = (job_index + 1) * env.n_machines + 1
            remaining_duration = sum([env.G.nodes[i]['duration'] for i in range(start_at, stop_at)])
            actions[job_index] = remaining_duration

        a = int(np.argmin(actions))
        n_state, reward, done, info = env.step(a)

    end = time.perf_counter()
    solving_duration = end - start

    if plot_results:
        env.render(show=["gantt_console"])
        log.info(f"solving duration: {solving_duration:2f} sec")

    makespan = info["makespan"]
    info["solving_duration"] = solving_duration
    info["GSRPT_solving_duration"] = solving_duration
    info["GSRPT_makespan"] = solving_duration

    return makespan, info


if __name__ == '__main__':
    import jss_utils.jsp_or_tools_solver as or_solver

    jsp = parser.get_instance_by_name("ft06")

    optimal_makespan, *_ = or_solver.solve_jsp(jsp_instance=jsp, plot_results=False)
    makespan, *_ = solve_jsp(jsp_instance=jsp)

    log.info(f"makespan to optimal makespan ratio: {makespan / optimal_makespan:.2f}")
