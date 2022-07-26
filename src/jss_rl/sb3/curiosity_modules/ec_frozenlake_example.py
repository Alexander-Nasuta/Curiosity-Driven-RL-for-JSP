from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvWrapper

from jss_rl.sb3.curiosity.ec import EpisodicCuriosityModuleWrapper
from jss_rl.sb3.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_rl.sb3.moving_avarage import MovingAverage


def ec_frozenlake_example(total_timesteps: int = 10_000, n_envs: int = 1) -> None:
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
        n_envs=n_envs
    )

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

    venv = DistanceWrapper(venv=venv)

    ec_venv = EpisodicCuriosityModuleWrapper(
        venv=venv,
        **{
            "alpha": 1.0,
            "beta": 0.5,
            "gamma": 2,
            "embedding_dim": 64,
            "episodic_memory_capacity": 8,
        }
    )
    ec_venv = VecMonitor(venv=ec_venv)
    ec_model = PPO('MlpPolicy', ec_venv, verbose=0)
    ec_model.learn(total_timesteps=budget)


if __name__ == '__main__':
    ec_frozenlake_example()