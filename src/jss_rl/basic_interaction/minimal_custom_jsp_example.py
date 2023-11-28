import sys

import numpy as np
import jss_utils.jsp_or_tools_solver as or_tools_solver
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv
from jss_graph_env.disjunctive_graph_jss_visualizer import DisjunctiveGraphJspVisualizer
import matplotlib.pyplot as plt


def ma_jsp_example() -> None:
    jsp = np.array([
        [
            [0, 1, 2, 3],  # job 0 (engineer’s hammer)
            [0, 2, 1, 3]  # job 1  (Nine Man Morris)
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4]  # task durations of job 1
        ]

    ])

    _, _, df, _ = or_tools_solver.solve_jsp(jsp_instance=jsp, plot_results=True)
    print(df.to_dict())
    sys.exit(1)

    c_map = plt.cm.get_cmap("jet")  # select the desired cmap
    arr = np.linspace(0, 1, 4)  # create a list with numbers from 0 to 1 with n items
    colors = {f"Machine {resource}": c_map(val) for resource, val in enumerate(arr)}

    jarr = np.linspace(0, 1, 2)  # create a list with numbers from 0 to 1 with n items
    jcolors = {f"Job {resource}": c_map(val) for resource, val in enumerate(jarr)}
    print(jcolors)

    print(df)
    print(colors)
    visualizer = DisjunctiveGraphJspVisualizer()
    #visualizer.render_gantt_in_window(df=df, colors=colors, wait=None)
    latex_code = visualizer.latex_tikz_figure_gantt(df=df, colors=colors)
    print(latex_code)
    env = DisjunctiveGraphJssEnv(jps_instance=jsp, normalize_observation_space=True, flat_observation_space=False)
    # env.render(show=["graph_window"], wait=None)
    for s in [4, 0,
              #0, 1, 6,
              #2,
              #3, 7
              ]:
        obs, *_ = env.step(s)
        env.render()
    print(obs)
    env.render(show=["gantt_console", "graph_console", "gantt_window"], wait=None)


if __name__ == '__main__':
    ma_jsp_example()
