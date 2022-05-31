
import time

import jss_utils.jsp_env_utils as env_utils

from jss_utils.jss_logger import log


if __name__ == '__main__':

    env = env_utils.get_pre_configured_example_env()

    done = False
    score = 0

    iteration_count = 0
    start = time.perf_counter()
    for i in range(env.total_tasks_without_dummies):
        n_state, reward, done, info = env.step(i)
        score += reward
        iteration_count += 1

    end = time.perf_counter()

    env.render(show=["gantt_console", "graph_console"])

    log.info(f"score: {score}")
    total_duration = end - start
    log.info(f"total duration: {total_duration:2f} sec")
    log.info(f"average iteration duration: {total_duration / iteration_count:4f} sec")
