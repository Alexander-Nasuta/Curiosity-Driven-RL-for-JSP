from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvWrapper

from jss_rl.sb3.curiosity_modules.icm_wrapper import IntrinsicCuriosityModuleWrapper
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_rl.sb3.util.moving_avarage import MovingAverage


class DistanceWrapper(VecEnvWrapper):

    def __init__(self, venv):
        self.distances = MovingAverage(capacity=1000)
        VecEnvWrapper.__init__(self, venv=venv)
        self._steps = 0

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()
        self._steps += self.venv.num_envs

        for i, o in enumerate(observations):
            x, y = o % 8, o // 8  # frozen lake with 8x8 size
            distance_from_origin = (x ** 2 + y ** 2) ** 0.5
            self.distances.add(distance_from_origin)
            print(f"[{self._steps}] distance_from_origin: {distance_from_origin:.4f},"
                  f" moving avarage distance_from_origin: {self.distances.mean():.4f}")

        return observations, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        observations = self.venv.reset()
        return observations


def icm_frozenlake_example(total_timesteps: int = 10_000) -> None:
    budget = total_timesteps

    env_name = "FrozenLake-v1"
    env_kwargs = {
        "desc": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFF",
            "FFFFFFFG",
        ],
        "is_slippery": False,
    }

    venv = make_vec_env_without_monitor(
        env_id=env_name,
        env_kwargs=env_kwargs,
        wrapper_class=TimeLimit,
        wrapper_kwargs={"max_episode_steps": 16},
        n_envs=1
    )

    venv = DistanceWrapper(venv=venv)

    icm_venv = IntrinsicCuriosityModuleWrapper(
        venv=venv,
        exploration_steps=int(0.5 * budget),
        feature_net_hiddens=[],
        forward_fcnet_net_hiddens=[256],
        inverse_feature_net_hiddens=[256],

        maximum_sample_size=16,

        clear_memory_on_end_of_episode=True,
        postprocess_on_end_of_episode=True,

        clear_memory_every_n_steps=None,
        postprocess_every_n_steps=None,
    )
    icm_venv = VecMonitor(venv=icm_venv)
    icm_model = PPO('MlpPolicy', icm_venv, verbose=0)
    icm_model.learn(total_timesteps=budget)


if __name__ == '__main__':
    icm_frozenlake_example()
