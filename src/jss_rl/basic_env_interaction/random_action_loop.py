import numpy as np
import jss_utils.jsp_or_tools_solver as or_solver

from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

if __name__ == '__main__':

    jsp = np.array([
        [
            [1, 2, 0],  # job 0
            [0, 2, 1]  # job 1
        ],
        [
            [17, 12, 19],  # task durations of job 0
            [8, 6, 2]  # task durations of job 1
        ]

    ])

    # calc or specify lower-bound/scaling_divisor. using google or tools for calculation here
    optimal_makespan, *_ = or_solver.solve_jsp(jsp_instance=jsp, plot_results=False)

    env = DisjunctiveGraphJssEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        scaling_divisor=optimal_makespan,
        scale_reward=True,
        normalize_observation_space=True,
        flat_observation_space=True,
        action_mode='task',  # alternatve 'job'
        dtype='float32'
    )

    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        # note: gantt_window and graph_window use a lot of resources
        # env.render(show=["gantt_console", "gantt_window", "graph_console", "graph_window"])
        env.render()
        score += reward

    env.render(wait=1_000)  # render window closes automatically after 1 seconds
    # env.render(wait=None) # render window closes when any button is pressed (when the render window is focused)
