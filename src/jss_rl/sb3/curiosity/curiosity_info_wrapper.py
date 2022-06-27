from typing import Union

from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

import numpy as np


class CuriosityInfoWrapper(VecEnvWrapper):

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        observations = self.venv.reset()

        self.sub_env_stats = [{
            **self.sub_env_stats[i],
            "extrinsic_rewards": [],
            "intrinsic_rewards": [],
        } for i in range(self.venv.num_envs)]
        return observations

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()

        self.global_stats["n_total_episodes"] += dones.sum()
        self.global_stats["n_postprocessings"] = self.n_postprocessings
        self.global_stats["_num_timesteps"] = self._num_timesteps

        intrinsic_rewards = np.zeros(self.venv.num_envs)
        augmented_rewards = rewards + intrinsic_rewards

        extended_infos = [info.copy() for info in infos]

        for i in range(self.venv.num_envs):
            self.sub_env_stats[i]["extrinsic_rewards"].append(rewards[i])
            self.sub_env_stats[i]["intrinsic_rewards"].append(intrinsic_rewards[i])

            if dones[i]:
                self.sub_env_stats[i]["n_sub_env_episodes"] += 1

                extended_infos[i]["extrinsic_return"] = sum(self.sub_env_stats[i]["extrinsic_rewards"])
                self.sub_env_stats[i]["extrinsic_rewards"] = []

                extended_infos[i]["intrinsic_return"] = sum(self.sub_env_stats[i]["intrinsic_rewards"])
                self.sub_env_stats[i]["intrinsic_rewards"] = []

                extended_infos[i]["bonus_return"] = extended_infos[i]["intrinsic_return"]
                extended_infos[i]["total_return"] = \
                    extended_infos[i]["intrinsic_return"] + extended_infos[i]["extrinsic_return"]

            extended_infos[i]["extrinsic_reward"] = rewards[i]
            extended_infos[i]["intrinsic_reward"] = intrinsic_rewards[i]
            extended_infos[i]["bonus_reward"] = intrinsic_rewards[i]
            extended_infos[i]["total_reward"] = augmented_rewards[i]
            extended_infos[i]["n_total_episodes"] = self.global_stats["n_total_episodes"]
            extended_infos[i]["n_postprocessings"] = self.global_stats["n_postprocessings"]
            extended_infos[i]["_num_timesteps"] = self.global_stats["_num_timesteps"]



        return observations, augmented_rewards, dones, extended_infos

    def step_async(self, actions: np.ndarray) -> None:
        self._num_timesteps += self.venv.num_envs  # one step per env
        self.venv.step_async(actions)

    def __init__(self, venv: Union[VecEnvWrapper, VecEnv, DummyVecEnv]):
        VecEnvWrapper.__init__(self, venv=venv)

        #
        self.n_postprocessings = 0
        self._num_timesteps = 0

        # statistics
        self.global_stats = {
            "n_total_episodes": 0,
            "n_postprocessings": self.n_postprocessings,
        }
        self.sub_env_stats = [{
            "extrinsic_rewards": [],
            "intrinsic_rewards": [],
            "n_sub_env_episodes": 0,
        } for _ in range(self.venv.num_envs)]



if __name__ == '__main__':
    from gym.wrappers import TimeLimit
    from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
    from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper, DummyVecEnv
    from stable_baselines3 import A2C, PPO

    print("##### CartPole-v1 #####")
    budget = 10_000
    eval_episodes = 10
    env_id = "CartPole-v1"

    venv = make_vec_env_without_monitor(
        env_id=env_id,
        env_kwargs={},
        n_envs=4
    )
    cartpole_venv = VecMonitor(venv=venv)
    # model1 = A2C('MlpPolicy', cartpole_venv, verbose=0, seed=773)
    # model1.learn(total_timesteps=budget)
    # mean_reward, std_reward = evaluate_policy(model1, cartpole_venv, n_eval_episodes=eval_episodes)
    # print(f"without icm: {mean_reward=}, {std_reward=}")
    cartpole_venv.reset()
    cartpole_venv = CuriosityInfoWrapper(
        venv=venv,
    )

    icm_model = PPO('MlpPolicy', cartpole_venv, verbose=0)
    icm_model.learn(total_timesteps=budget)