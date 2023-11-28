import gym

import wandb as wb

import jss_utils.PATHS as PATHS

from rich.progress import Progress

from jss_rl.sb3.experiments.jss.jsp_6x6.jss_experiment_ec_6x6 import run_ppo_ec_jss_experiment_6x6
from jss_rl.sb3.experiments.jss.jsp_6x6.jss_experiment_icm_6x6 import run_ppo_icm_jss_experiment_6x6
from jss_rl.sb3.experiments.jss.jsp_6x6.jss_experiment_plain_ppo_6x6 import run_ppo_jss_experiment_6x6
from jss_utils.jss_logger import log
from jss_utils.name_generator import generate_name


def run_experiment_series_6x6(total_timesteps: int, instance_name: str, wb_project: str, num_runs_per_module: int = 10) \
        -> None:
    name = generate_name()
    additional_config = {
        "series_id": 'Purple_Lemon_95_ed306729'
        # "series_id": name
    }
    log.info(f"running an experiment series ('{name}') on 'GraphJsp-v0' environment.")

    with Progress() as progress:
        #task1 = progress.add_task("[cyan]PPO", total=num_runs_per_module)
        task2 = progress.add_task("[cyan]PPO + ICM", total=num_runs_per_module)
        task3 = progress.add_task("[cyan]PPO + EC", total=num_runs_per_module)

        for algo, task, experiment_function in zip(
                [
                    #"PPO",
                    "PPO + ICM",
                    "PPO + EC"
                ],
                [
                    #task1,
                    task2,
                    task3
                ],
                [
                    #run_ppo_jss_experiment_6x6,
                    run_ppo_icm_jss_experiment_6x6,
                    run_ppo_ec_jss_experiment_6x6
                ]
        ):
            log.info(f"running experiments with {algo} algorithm.")
            for _ in range(num_runs_per_module):
                # noinspection PyArgumentList
                experiment_function(
                    total_timesteps=total_timesteps,
                    project=wb_project,
                    instance_name=instance_name,
                    additional_config=additional_config
                )
                progress.update(task, advance=1)


if __name__ == '__main__':
    gym.envs.register(
        id='GraphJsp-v0',
        entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
        kwargs={},
    )
    wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))
    # Purple_Lime_30_b5f4822e (wrong icm parametrisation, 150_000 steps)
    # Purple_Lemon_95_ed306729 (fixed parameters 500_000 steps)
    run_experiment_series_6x6(
        total_timesteps=500_000,
        instance_name="ft06",
        wb_project="MA-nasuta",
        num_runs_per_module=3
    )
