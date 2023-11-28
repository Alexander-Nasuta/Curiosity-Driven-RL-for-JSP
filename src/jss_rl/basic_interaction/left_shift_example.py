import numpy as np
import jss_utils.jsp_or_tools_solver as or_tools_solver
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv
from jss_graph_env.disjunctive_graph_jss_visualizer import DisjunctiveGraphJspVisualizer
import matplotlib.pyplot as plt


def ma_left_shift_jsp_example() -> None:
    jsp = np.array([
        [
            [0, 1, 2],  # job 0
            [2, 0, 1],  # job 1
            [0, 2, 1]  # job 3
        ],
        [
            [1, 1, 5],  # task durations of job 0
            [5, 3, 3],  # task durations of job 1
            [3, 6, 3]  # task durations of job 1
        ]

    ])
    # _, _, df, _ = or_tools_solver.solve_jsp(jsp_instance=jsp, plot_results=True)

    env = DisjunctiveGraphJssEnv(jps_instance=jsp, perform_left_shift_if_possible=False, action_mode='job')
    # env.render(show=["graph_window"], wait=None)
    for s in [0, 1, 0, 1, 2]:
        env.step(s)
        #env.render()
    env.render(show=["gantt_console", "graph_console"], wait=None)
    df = env.network_as_dataframe()
    print(df)


if __name__ == '__main__':
    ma_left_shift_jsp_example()
