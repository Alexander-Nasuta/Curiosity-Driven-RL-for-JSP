from collections import deque
from statistics import mean
from typing import List

import wandb as wb
import numpy as np

from types import ModuleType

from stable_baselines3.common.callbacks import BaseCallback


class FrozenlakeLoggerCallBack(BaseCallback):

    def __init__(self, num_envs: int, wandb_ref: ModuleType = wb, verbose=0):
        super(FrozenlakeLoggerCallBack, self).__init__(verbose)

        self.wandb_ref = wandb_ref
        self.num_envs = num_envs

        self.max_distance = 0.0
        self.visited_states = set()
        self.visited_state_action_pairs = set()
        self.max_return = 0.0

        self.temp_trajectories = [
            deque(maxlen=3 * 16) for _ in range(self.num_envs)  # num envs
        ]
        self.best_trajectory = None

        self._n_rollouts = 0

        self.log_fields = [
            "extrinsic_return",
            "intrinsic_return",
            "bonus_return",
            "total_return",
            "n_postprocessings",
            "n_total_episodes",
            "_num_timesteps",
            "distance_from_origin",

            "loss",
            "inverse_loss",
            "forward_loss",
        ]

    def _get_vals(self, field: str) -> List:
        return [env_info[field] for env_info in self.locals['infos'] if field in env_info.keys()]

    def _on_step(self) -> bool:
        # self.num_timesteps
        for i, obs in enumerate(self.locals["obs_tensor"]):
            self.temp_trajectories[i].append(obs)

        self.max_distance = max(self.max_distance, *self._get_vals("distance_from_origin"))

        new_max = max(0.0, self.max_return, *self._get_vals("total_return"))
        if new_max >= self.max_return:
            self.max_return = new_max
            max_i = np.argmax(np.array(self._get_vals("distance_from_origin")))
            self.best_trajectory = list(self.temp_trajectories[max_i])

        self.visited_states = self.visited_states.union([obs.item() for obs in self.locals["obs_tensor"]])

        self.visited_state_action_pairs = self.visited_state_action_pairs.union(
            [(obs.item(), actions) for obs, actions in zip(self.locals["obs_tensor"], self.locals["actions"])]
        )

        # note num_timesteps increments always in num_envs (default 8)
        if self.num_timesteps and self.num_timesteps % (self.num_envs * 1250) == 0:
            tab = self.wandb_ref.Table(
                columns=[" ", *[f"step_{i}" for i in range(len(self.temp_trajectories[0]))]],
                data=[
                    ["state", *[elem.item() for elem in self.temp_trajectories[0]]],
                    ["state (index 1)", *[elem.item() + 1 for elem in self.temp_trajectories[0]]],
                    ["col", *[elem.item() % 8 for elem in self.temp_trajectories[0]]],
                    ["row", *[elem.item() // 8 for elem in self.temp_trajectories[0]]],
                ]
            )
            self.wandb_ref.log({
                f"trajectory_{self.num_timesteps}steps_env_0": tab,
                "num_timesteps": self.num_timesteps
            })

        if self.wandb_ref:
            self.wandb_ref.log({
                **{f: mean(self._get_vals(f)) for f in self.log_fields if self._get_vals(f)},
                "max_distance": self.max_distance,
                "n_visited_states": len(self.visited_states),
                "n_visited_state_action_pairs": len(self.visited_state_action_pairs),
                "explored_states": len(self.visited_states) / (8 * 8),
                # no actions in terminal state
                "explored_state_action_pairs": len(self.visited_state_action_pairs) / (8 * 8 * 4 - 4),
                "num_timesteps": self.num_timesteps
            })

        return True

    def _on_rollout_end(self) -> None:
        self._n_rollouts += 1
        tab = self.wandb_ref.Table(
            columns=[" ", *[f"step_{i}" for i in range(len(self.temp_trajectories[0]))]],
            data=[
                ["state", *[elem.item() for elem in self.temp_trajectories[0]]],
                ["state (index 1)", *[elem.item() + 1 for elem in self.temp_trajectories[0]]],
                ["col", *[elem.item() % 8 for elem in self.temp_trajectories[0]]],
                ["row", *[elem.item() // 8 for elem in self.temp_trajectories[0]]],
            ]
        )
        self.wandb_ref.log({
            f"rollout_{self._n_rollouts}_sample_trajectory": tab,
            "num_timesteps": self.num_timesteps
        })

    def _on_training_end(self) -> None:
        tab = self.wandb_ref.Table(
            columns=[" ", *[f"step_{i}" for i in range(len(self.best_trajectory))]],
            data=[
                ["state", *[elem.item() for elem in self.best_trajectory]],
                ["state (index 1)", *[elem.item() + 1 for elem in self.best_trajectory]],
                ["col", *[elem.item() % 8 for elem in self.best_trajectory]],
                ["row", *[elem.item() // 8 for elem in self.best_trajectory]],
            ]
        )
        self.wandb_ref.log({
            "highest_score_trajectory": tab,
            "num_timesteps": self.num_timesteps
        })
