import gym

import wandb as wb

import jss_utils.PATHS as PATHS

from rich.progress import Progress

from jss_rl.sb3.experiments.jss.jsp_10x10.jss_experiment_icm_10x10 import run_ppo_icm_jss_experiment_10x10
from jss_rl.sb3.experiments.jss.jsp_10x10.jss_experiment_plain_ppo_10x10 import run_ppo_jss_experiment_10x10
from jss_utils.jss_logger import log
from jss_utils.name_generator import generate_name


def run_experiment_series_10x10(total_timesteps: int, instance_name: str, wb_project: str,
                                num_runs_per_module: int = 1) -> None:
    name = generate_name()
    additional_config = {
        # "series_id": 'Vantablack_Lemon_45_db056561'
        # "series_id": "Green_Coconut_16_5e3e9a40"
         "series_id": name
    }
    log.info(f"running an experiment series ('{name}') on 'GraphJsp-v0' environment.")

    with Progress() as progress:
        task1 = progress.add_task("[cyan]PPO", total=num_runs_per_module)
        task2 = progress.add_task("[cyan]PPO + ICM", total=num_runs_per_module)

        for algo, task, experiment_function in zip(
                [
                    "PPO",
                    "PPO + ICM"
                ],
                [
                    task1,
                    task2
                ],
                [
                   run_ppo_jss_experiment_10x10,
                   run_ppo_icm_jss_experiment_10x10
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
    # Vantablack_Lemon_45_db056561 # kind_sweep_18
    # Green_Coconut_16_5e3e9a40 # kind_sweep_18
    # Yellow_Passionfruit_31_986cdd34 #still-sweep-27
    run_experiment_series_10x10(
        total_timesteps=4_000_000,
        instance_name="ft10",
        wb_project="MA-nasuta",
        num_runs_per_module=1
    )
