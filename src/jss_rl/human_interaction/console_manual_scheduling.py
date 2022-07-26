import inquirer

from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv
from jss_utils.jss_logger import log

import jss_utils.jsp_env_utils as env_utils

if __name__ == '__main__':

    jsp, lb = env_utils.get_benchmark_instance_and_lower_bound(name="ft06")

    env = DisjunctiveGraphJssEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        scaling_divisor=lb,
        scale_reward=True,
        normalize_observation_space=True,
        flat_observation_space=True,
        action_mode="task",
        verbose=1,

        default_visualisations=["gantt_console", "graph_console"]
    )

    done = False

    log.info("each task/node corresponds to an action")

    while not done:
        env.render()
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

    log.info("the JSP is completely scheduled.")
