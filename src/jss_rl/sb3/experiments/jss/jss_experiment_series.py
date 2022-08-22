import gym

import wandb as wb

import jss_utils.PATHS as PATHS

from rich.progress import Progress

from jss_rl.sb3.experiments.jss.jss_experiment_ec import run_ppo_ec_jss_experiment
from jss_rl.sb3.experiments.jss.jss_experiment_icm import run_ppo_icm_jss_experiment
from jss_rl.sb3.experiments.jss.jss_experiment_plain_ppo import run_ppo_jss_experiment
from jss_utils.jss_logger import log
from jss_utils.name_generator import generate_name


def run_experiment_series(total_timesteps: int, instance_name: str, wb_project: str, num_runs_per_module: int = 10) \
        -> None:
    name = generate_name()
    additional_config = {
        "series_id": name
    }
    log.info(f"running an experiment series ('{name}') on 'GraphJsp-v0' environment.")

    with Progress() as progress:
        task1 = progress.add_task("[cyan]PPO", total=num_runs_per_module)
        task2 = progress.add_task("[cyan]PPO + ICM", total=num_runs_per_module)
        task3 = progress.add_task("[cyan]PPO + EC", total=num_runs_per_module)

        for algo, task, experiment_function in zip(
                ["PPO", "PPO + ICM", "PPO + EC"],
                [task1, task2, task3],
                [run_ppo_jss_experiment, run_ppo_icm_jss_experiment, run_ppo_ec_jss_experiment]
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
    run_experiment_series(
        total_timesteps=120_000,
        instance_name="ft06",
        wb_project="test",
        num_runs_per_module=1
    )
