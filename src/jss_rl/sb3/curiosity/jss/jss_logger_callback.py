from collections import deque
from statistics import mean
from typing import List

import wandb as wb
import torch

from types import ModuleType

from stable_baselines3.common.callbacks import BaseCallback


class JssLoggerCallback(BaseCallback):

    def __init__(self, wandb_ref: ModuleType = wb, verbose=0):
        super(JssLoggerCallback, self).__init__(verbose)

        self.wandb_ref = wandb_ref

        self.visited_states = set()
        self.visited_state_action_pairs = set()

        self.total_left_shifts = 0
        self.last_100_left_shifts = deque(maxlen=100)

        self.senv_fields = [
            "extrinsic_return",
            "intrinsic_return",
            "bonus_return",
            "total_return",
            "left_shift",

            "makespan",
        ]
        self.venv_fields = [
            "n_postprocessings",
            "n_total_episodes",
            "_num_timesteps",

            "loss",
            "inverse_loss",
            "forward_loss",

            "scaling_divisor",
        ]

    def _get_vals(self, field: str) -> List:
        return [env_info[field] for env_info in self.locals['infos'] if field in env_info.keys()]

    def _on_step(self) -> bool:

        self.visited_states = self.visited_states.union(
            [tuple(torch.ravel(obs).tolist()) for obs in self.locals["obs_tensor"]])

        self.visited_state_action_pairs = self.visited_state_action_pairs.union(
            [(tuple(torch.ravel(obs).tolist()), action) for obs, action in
             zip(self.locals["obs_tensor"], self.locals["actions"])]
        )

        ls_list = self._get_vals("left_shift")
        if len(ls_list):
            self.total_left_shifts += sum(ls_list)
            self.last_100_left_shifts.extend(ls_list)

        if self.wandb_ref:
            self.wandb_ref.log({
                **{
                    f"{f}_env_{i}": info[f]
                    for i, info in enumerate(self.locals['infos'])
                    for f in self.senv_fields
                    if f in info.keys()
                },
                **{f"{f}_mean": mean(self._get_vals(f)) for f in self.senv_fields if self._get_vals(f)},
                **{f"{f}": mean(self._get_vals(f)) for f in self.venv_fields if self._get_vals(f)},
                "n_visited_states": len(self.visited_states),
                "n_visited_state_action_pairs": len(self.visited_state_action_pairs),
                "total_left_shifts": self.total_left_shifts,
                "left_shift_moving_avg_pct": sum(self.last_100_left_shifts) / len(self.last_100_left_shifts) * 100,
                "left_shift_pct": self.total_left_shifts / self.num_timesteps * 100,
                "num_timesteps": self.num_timesteps
            })

        return True
