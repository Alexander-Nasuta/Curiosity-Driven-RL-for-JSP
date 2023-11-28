import inquirer
import numpy as np

from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv
from jss_utils.jss_logger import log



if __name__ == '__main__':

    #env = env_utils.get_pre_configured_example_env(name="ft06")
    #jsp, _ = env_utils.get_benchmark_instance_and_details(name="ft06")
    #print(repr(jsp))


    jsp = np.array([
        [
            [0, 1, 2, 3],  # job 0 (engineer’s hammer)
            [0, 2, 1, 3],  # job 1  (Nine Man Morris)
        ],
        [
            [11, 3, 3, 12],  # task durations of job 0
            [5, 16, 7, 4],  # task durations of job 1
        ]

    ])

    env = DisjunctiveGraphJssEnv(jps_instance=jsp, scaling_divisor=40.0,)

    done = False
    log.info("each task/node corresponds to an action")

    while not done:
        env.render(
            show=["gantt_console", "gantt_window", "graph_console", "graph_window"],
            #,stack='vertically'
        )
        questions = [
            inquirer.List(
                "task",
                message="Which task should be scheduled next?",
                choices=[
                    (f"Task {task_id}", task_id)
                    for task_id, bol in enumerate(env.valid_action_mask(), start=1)
                    if bol
                ],
            ),
        ]
        action = inquirer.prompt(questions)["task"] - 1  # note task are index 1 in the viz, but index 0 in action space
        n_state, reward, done, info = env.step(action)
        # note: gantt_window and graph_window use a lot of resources

    log.info(f"the JSP is completely scheduled.")
    log.info(f"makespan: {info['makespan']}")
    log.info("press any key to close the window (while the window is focused).")
    # env.render(wait=None)  # wait for keyboard input before closing the render window
    env.render(
        wait=None,
        show=["gantt_console", "graph_console", "graph_window"],
        #stack='vertically'
    )
