from statistics import mean
from typing import List

from stable_baselines3.common.callbacks import BaseCallback

from jss_rl.sb3.moving_avarage import MovingAverage


class InfoFieldMovingAverageLogger(BaseCallback):

    def __init__(self,
                 fields: List[str],
                 field_capacities: List[int],
                 wandb_ref=None,
                 verbose=0):
        super(InfoFieldMovingAverageLogger, self).__init__(verbose)
        self.log_fields = fields
        self.wandb_ref = wandb_ref

        if field_capacities is None:
            field_capacities = [100] * len(fields)

        if len(fields) != len(field_capacities):
            raise ValueError(f"`fields` and `field_capacities` must be the same length.")

        self.memory = {
            field: MovingAverage(capacity=cap)
            for field, cap in zip(fields, field_capacities)
        }

    def _on_step(self) -> bool:
        # env_infos of the envs inside the venv that are done with the episode
        for env_info in self.locals['infos']:
            for field in self.log_fields:
                if field not in env_info.keys():
                    continue
                val = env_info[field]
                self.memory[field].add(val)

        if self.wandb_ref:
            for field in self.log_fields:
                vals = [ev[field] for ev in self.locals['infos'] if field in ev.keys()]
                if len(vals):
                    mean_val = mean(vals)
                    self.wandb_ref.log({
                        field: mean_val,
                        "num_timesteps": self.num_timesteps
                    })
        return True

    def _on_rollout_end(self) -> bool:
        for field in self.log_fields:
            self.logger.record(f"rollout/{field}", self.memory[field].mean())
            if self.wandb_ref:
                self.wandb_ref.log({
                    f"rollout/{field}_mean": self.memory[field].mean(),
                    "num_timesteps": self.num_timesteps
                })
        return True
