import numpy as np

from typing import List

from stable_baselines3.common.callbacks import BaseCallback


# for num vals present on every step
class WB_InfoLoggerCallback(BaseCallback):

    def _on_step(self) -> bool:
        for f in self.fields:
            vals = np.array([self.locals['infos'][i][f] for i in range(self.n_envs)])
            mean = np.mean(vals)
            self.rollout_memory[f] = np.vstack((self.rollout_memory[f], vals))
            moving_average = np.mean(
                self.rollout_memory[f][-self.moving_average_size:],
                axis=0
            )


            for i in range(self.n_envs):
                self.wandb_ref.log({
                    f"env_{i}/{f}": self.locals['infos'][i][f],
                    f"env_{i}/{f}_moving_avarage_{self.moving_average_size}": moving_average[i],
                    "num_timesteps": self.num_timesteps
                })

            self.wandb_ref.log({
                f"envs_mean/{f}": mean,
                "num_timesteps": self.num_timesteps
            })

            self.wandb_ref.log({
                f"envs_mean/{f}_min": self.rollout_memory[f].min(),
                f"envs_mean/{f}_max": self.rollout_memory[f].max(),
                "num_timesteps": self.num_timesteps
            })

        return True

    def __init__(self, fields: List[str], n_envs: int, moving_average_size: int = 100, wandb_ref=None, verbose=0):
        super(WB_InfoLoggerCallback, self).__init__(verbose=verbose)
        self.wandb_ref = wandb_ref
        self.fields = fields
        self.n_envs = n_envs
        self.moving_average_size = moving_average_size
        self.rollout_memory = {
            field: np.empty(
                shape=(0, self.n_envs)
            ) for field in self.fields
        }

    def _on_rollout_start(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        for f in self.fields:
            rollout_means = np.mean(self.rollout_memory[f], axis=0)
            rollout_maxes = np.max(self.rollout_memory[f], axis=0)
            rollout_mins = np.min(self.rollout_memory[f], axis=0)
            for i in range(self.n_envs):
                self.wandb_ref.log({
                    f"env_{i}/rollout/{f}_mean": rollout_means[i],
                    f"env_{i}/rollout/{f}_max": rollout_maxes[i],
                    f"env_{i}/rollout/{f}_min": rollout_mins[i],
                    "num_timesteps": self.num_timesteps
                })

        self.rollout_memory = {
            field: np.empty(
                shape=(0, self.n_envs)
            ) for field in self.fields
        }
