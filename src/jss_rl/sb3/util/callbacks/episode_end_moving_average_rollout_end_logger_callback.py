from statistics import mean
from typing import List, Callable

from stable_baselines3.common.callbacks import BaseCallback

from jss_rl.sb3.util.moving_avarage import MovingAverage


class EpisodeEndMovingAverageRolloutEndLoggerCallback(BaseCallback):

    def __init__(self, fields: List[str], capacity: int = 100, wandb_ref=None, verbose=0):
        super(EpisodeEndMovingAverageRolloutEndLoggerCallback, self).__init__(verbose)
        self.log_fields = fields
        self.capacity = capacity
        self.wandb_ref = wandb_ref
        self.memory = {
            field: MovingAverage(capacity=self.capacity) for field in fields
        }

    def _on_step(self) -> bool:
        # env_infos of the envs inside the venv that are done with the episode
        env_infos = [self.locals['infos'][i] for i, done in enumerate(self.locals['dones']) if done]
        for env_info in env_infos:
            for field in self.log_fields:
                val = env_info[field]
                self.memory[field].add(val)

        if self.wandb_ref and len(env_infos):
            for field in self.log_fields:
                vals = [
                    ev[field] for ev in self.locals['infos']
                    if field in ev.keys()
                ]
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
                    f"rollout2/{field}": self.memory[field].mean(),
                    "num_timesteps": self.num_timesteps
                })

        return True
