from typing import List, Callable

from stable_baselines3.common.callbacks import BaseCallback

from jss_rl.sb3.util.moving_avarage import MovingAverage


class EpisodeEndMovingAverageRolloutEndLoggerCallback(BaseCallback):

    def __init__(self, fields: List[str], capacity: int = 100, verbose=0):
        super(EpisodeEndMovingAverageRolloutEndLoggerCallback, self).__init__(verbose)
        self.log_fields = fields
        self.capacity = capacity
        self.memory = {field: MovingAverage(capacity=self.capacity) for field in fields}

    def _on_step(self) -> bool:
        # env_infos of the envs inside the venv that are done with the episode
        env_infos = [self.locals['infos'][i] for i, done in enumerate(self.locals['dones']) if done]
        for env_info in env_infos:
            for field in self.log_fields:
                val = env_info[field]
                self.memory[field].add(val)
        return True

    def _on_rollout_end(self) -> bool:
        for field in self.log_fields:
            self.logger.record(f"rollout/{field}", self.memory[field].mean())
        return True
