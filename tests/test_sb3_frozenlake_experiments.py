import wandb as wb
import jss_utils.PATHS as PATHS
from jss_rl.sb3.experiments.frozenlake.frozenlake_experiment_series import run_experiment_series


def test_frozenlake_experiments():
    wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))
    run_experiment_series(
        total_timesteps=1_000,
        wb_project="test",
        num_runs_per_module=1
    )
