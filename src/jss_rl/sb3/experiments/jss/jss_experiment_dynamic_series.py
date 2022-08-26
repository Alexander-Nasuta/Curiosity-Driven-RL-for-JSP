import gym

import wandb as wb

import jss_utils.PATHS as PATHS

from rich.progress import Progress

from jss_rl.sb3.experiments.jss.jss_experiment_dynamic_icm import run_dynamic_ppo_icm_jss_experiment
from jss_rl.sb3.experiments.jss.jss_experiment_dynamic_ppo import run_dynamic_ppo_jss_experiment
from jss_utils.jss_logger import log
from jss_utils.name_generator import generate_name


def run_experiment_series(total_timesteps: int,
                          n_jobs: int,
                          n_machines: int,
                          eval_instance_name: str,
                          load_instance_every_n_rollouts: int,
                          wb_project: str,
                          pre_eval_learning_timesteps: int = 0,
                          num_runs_per_module: int = 10) \
        -> None:
    name = generate_name()
    additional_config = {
        "series_id": name
    }
    log.info(f"running an experiment series ('{name}') on 'GraphJsp-v0' environment.")

    with Progress() as progress:
        task1 = progress.add_task("[cyan]PPO", total=num_runs_per_module)
        task2 = progress.add_task("[cyan]PPO + ICM", total=num_runs_per_module)

        for algo, task, experiment_function in zip(
                ["PPO", "PPO + ICM"],
                [task1, task2],
                [run_dynamic_ppo_jss_experiment, run_dynamic_ppo_icm_jss_experiment]
        ):
            log.info(f"running experiments with {algo} algorithm.")
            for _ in range(num_runs_per_module):
                # noinspection PyArgumentList
                experiment_function(
                    total_timesteps=total_timesteps,
                    project=wb_project,
                    n_jobs=n_jobs,
                    n_machines=n_machines,
                    eval_instance_name=eval_instance_name,
                    load_instance_every_n_rollouts=load_instance_every_n_rollouts,
                    pre_eval_learning_timesteps=pre_eval_learning_timesteps,
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
