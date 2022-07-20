import numpy as np

from stable_baselines3.common.env_checker import check_env

from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv
from jss_utils.jsp_env_utils import get_pre_configured_example_env
from jss_utils.jsp_instance_details import download_benchmark_instances_details
from jss_utils.jsp_instance_downloader import download_instances


def test_env_checker():
    # if this code passes the env is most likely fine
    for left_shift in [True, False]:
        for normalize in [True, False]:
            for flat in [True, False]:
                for scale_rew in [True, False]:
                    for mode in ["job", "task"]:
                        for env_transform in [None, 'mask']:
                            for dt in ["float16", "float32", "float64"]:
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

                                env = DisjunctiveGraphJssEnv(
                                    jps_instance=jsp,
                                    perform_left_shift_if_possible=left_shift,
                                    scaling_divisor=None,
                                    scale_reward=scale_rew,
                                    normalize_observation_space=normalize,
                                    flat_observation_space=flat,
                                    action_mode=mode,
                                    dtype=dt,
                                    env_transform=env_transform,
                                    verbose=1
                                )

                                check_env(env)


def test_env_rl_loop():
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

    env = DisjunctiveGraphJssEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        scaling_divisor=None,
        scale_reward=True,
        normalize_observation_space=True,
        flat_observation_space=True,
        action_mode="task",
        verbose=1
    )

    done = False
    score = 0

    iteration_count = 0
    for i in range(env.total_tasks_without_dummies):
        n_state, reward, done, info = env.step(i)
        score += reward
        iteration_count += 1


def test_env_render():
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

    env = DisjunctiveGraphJssEnv(
        jps_instance=jsp,
        perform_left_shift_if_possible=True,
        scaling_divisor=None,
        scale_reward=True,
        normalize_observation_space=True,
        flat_observation_space=True,
        action_mode="task",
        verbose=1
    )

    env.render()


def test_naive_schedule():
    download_instances(start_id=1, end_id=2)
    download_benchmark_instances_details()
    env = get_pre_configured_example_env(name="abz5", action_mode="task")

    # done = False
    score = 0

    iteration_count = 0
    for i in range(env.total_tasks_without_dummies):
        n_state, reward, done, info = env.step(i)
        env.render(show=["gantt_window", "graph_console"])
        score += reward
        iteration_count += 1


