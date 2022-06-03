from typing import List, Callable

from stable_baselines3.common.callbacks import BaseCallback

from jss_rl.sb3.util.moving_avarage import MovingAverage


class InfoFieldMovingAvarageLogger(BaseCallback):

    def __init__(self,
                 fields: List[str],
                 field_capacities: List[int],
                 wandb_ref=None,
                 verbose=0):
        super(InfoFieldMovingAvarageLogger, self).__init__(verbose)
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
                    self.wandb_ref.log({field: val})
        return True

    def _on_rollout_end(self) -> bool:
        for field in self.log_fields:
            self.logger.record(f"rollout/{field}", self.memory[field].mean())
            if self.wandb_ref:
                self.wandb_ref.log({f"rollout/{field}": self.memory[field].mean()})
        return True
