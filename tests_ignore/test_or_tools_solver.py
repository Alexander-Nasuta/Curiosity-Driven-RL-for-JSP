import numpy as np

import jss_utils.jsp_or_tools_solver as or_tools_solver


def test_or_tools_solver():
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
    or_tools_solver.solve_jsp(jsp_instance=jsp, plot_results=True)

