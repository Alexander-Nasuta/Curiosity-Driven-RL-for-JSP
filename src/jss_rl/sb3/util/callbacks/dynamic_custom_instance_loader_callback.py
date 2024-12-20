import sys
from types import ModuleType

import jss_utils.jsp_env_utils as env_utils
import wandb as wb

from stable_baselines3.common.callbacks import BaseCallback

from jss_utils.jss_logger import log


class DynamicCustomInstanceLoaderCallback(BaseCallback):

    # only works with venv
    def __init__(self, load_instance_every_n_rollouts: int, verbose: int = 0, wandb_ref: ModuleType = None):
        super(DynamicCustomInstanceLoaderCallback, self).__init__(verbose=verbose)
        self._rollout_count = 0
        self._load_instance_every_n_rollouts = load_instance_every_n_rollouts
        self.wandb_ref = wandb_ref

    def _on_training_start(self) -> None:
        first_env = self.model.env.envs[0]
        n_jobs = first_env.n_jobs 
        n_machines = first_env.n_machines

        jsp, details, name = env_utils.get_random_custom_instance_and_details_and_name(
            n_jobs=n_jobs,
            n_machines=n_machines
        )

        log.info(f"loading custom instance '{name}' to venv.")
        if self.wandb_ref:
            self.wandb_ref.log({
                "jsp_instance": name,
                "num_timesteps": self.num_timesteps
            })

        for env in self.model.env.envs:
            env.load_instance(
                jsp_instance=jsp,
                scaling_divisor=details["lower_bound"]
            )

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self._load_instance_every_n_rollouts == 0:
            first_env = self.model.env.envs[0]
            n_jobs = first_env.n_jobs
            n_machines = first_env.n_machines

            jsp, details, name = env_utils.get_random_custom_instance_and_details_and_name(
                n_jobs=n_jobs,
                n_machines=n_machines
            )

            log.info(f"loading custom instance '{name}' to venv.")
            if self.wandb_ref:
                self.wandb_ref.log({
                    "jsp_instance": name,
                    "jsp_instance_details": details,
                    "num_timesteps": self.num_timesteps
                })

            for env in self.model.env.envs:
                env.load_instance(
                    jsp_instance=jsp,
                    scaling_divisor=details["lower_bound"]
                )

