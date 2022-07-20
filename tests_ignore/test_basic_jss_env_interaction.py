import numpy as np

from jss_rl.basic_interaction.random_action_loop import random_action_loop
from jss_rl.basic_interaction.trivial_schedule import trivial_schedule


def test_trivial_schedule(abz5):
    jsp_instance, jsp_instance_details = abz5
    lb = jsp_instance_details["lower_bound"]
    trivial_schedule(jsp_instance=jsp_instance, lower_bound=lb)

def test_random_action_loop():
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
    random_action_loop(jsp_instance=jsp)