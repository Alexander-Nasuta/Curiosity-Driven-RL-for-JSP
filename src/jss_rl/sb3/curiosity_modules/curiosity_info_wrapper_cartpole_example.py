from jss_rl.sb3.curiosity_modules.curiosity_info_wrapper import CuriosityInfoWrapper
from jss_rl.sb3.make_vec_env_without_monitor import make_vec_env_without_monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO


def curiosity_info_wrapper_example(total_timesteps: int = 10_000, n_envs: int = 4) -> None:
    budget = total_timesteps
    env_id = "CartPole-v1"

    venv = make_vec_env_without_monitor(
        env_id=env_id,
        env_kwargs={},
        n_envs=n_envs
    )
    cartpole_venv = VecMonitor(venv=venv)

    cartpole_venv.reset()
    cartpole_venv = CuriosityInfoWrapper(
        venv=venv,
    )

    icm_model = PPO('MlpPolicy', cartpole_venv, verbose=1)
    icm_model.learn(total_timesteps=budget)


if __name__ == '__main__':
    curiosity_info_wrapper_example()
