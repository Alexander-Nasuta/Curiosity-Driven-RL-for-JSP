import gym

import wandb as wb

import jss_utils.PATHS as PATHS

from rich.progress import Progress

from jss_rl.sb3.experiments.jss.jsp_6x6.jss_experiment_dynamic_icm_6x6 import run_dynamic_ppo_icm_jss_experiment_6x6
from jss_rl.sb3.experiments.jss.jsp_6x6.jss_experiment_dynamic_ppo_6x6 import run_dynamic_ppo_jss_experiment_6x6
from jss_utils.jss_logger import log
from jss_utils.name_generator import generate_name


def run_experiment_series_6x6(total_timesteps: int,
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
        "series_id": "Gray_Passionfruit_67_78a44583"
        #"series_id": name
    }
    log.info(f"running an experiment series ('{name}') on 'GraphJsp-v0' environment.")

    with Progress() as progress:
        #task1 = progress.add_task("[cyan]PPO", total=num_runs_per_module)
        task2 = progress.add_task("[cyan]PPO + ICM", total=num_runs_per_module)

        for algo, task, experiment_function in zip(
                [
                    #"PPO",
                    "PPO + ICM"
                ],
                [
                    #task1,
                    task2
                ],
                [
                    #run_dynamic_ppo_jss_experiment_6x6,
                    run_dynamic_ppo_icm_jss_experiment_6x6
                ]
        ):
            log.info(f"running experiments with {algo} algorithm.")
            for _ in range(num_runs_per_module):
                # noinspection PyArgumentList
                #  total_timesteps=50_000,
                #         n_machines=6,
                #         n_jobs=6,
                #         load_instance_every_n_rollouts=1,
                #         eval_instance_name="ft06",
                #         pre_eval_learning_timesteps=50_000,
                #         project="test"
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
    # NOTE: not sure with these might mixed them up with 6x6 single instance series.
    # check in wand before using them.
    #
    # Gray_Passionfruit_67_78a44583
    # Purple_Lime_30_b5f4822e
    # Purple_Lemon_95_ed306729
    run_experiment_series_6x6(
        total_timesteps=3_000_000,
        #total_timesteps=50_000,
        n_jobs=6,
        n_machines=6,
        eval_instance_name="ft06",
        load_instance_every_n_rollouts=1,
        wb_project="MA-nasuta",
        #wb_project="test",
        pre_eval_learning_timesteps=500_000,
        #pre_eval_learning_timesteps=10_000,
        num_runs_per_module=2
    )
