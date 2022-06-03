import inquirer

from jss_utils.jss_logger import log

import jss_utils.jsp_env_utils as env_utils

if __name__ == '__main__':

    env = env_utils.get_pre_configured_example_env()

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
    log.info("press any key to close the window (while the window is focused).")
    #env.render(wait=None)  # wait for keyboard input before closing the render window
    env.render(wait=5)
