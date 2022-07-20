from jss_rl.sb3.curiosity_modules.ec_wrapper import EpisodicCuriosityModuleWrapper
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
from stable_baselines3.common.vec_env import VecMonitor, VecEnvWrapper, DummyVecEnv
from stable_baselines3 import A2C, PPO


def ec_cartpole_example(total_timesteps: int = 10_000, n_envs: int = 4) -> None:
    budget = total_timesteps
    env_id = "CartPole-v1"

    venv = make_vec_env_without_monitor(
        env_id=env_id,
        env_kwargs={},
        n_envs=n_envs
    )
    cartpole_venv = VecMonitor(venv=venv)
    cartpole_venv.reset()
    cartpole_icm_venv = EpisodicCuriosityModuleWrapper(
        venv=venv,
    )

    ec_model = PPO('MlpPolicy', cartpole_icm_venv, verbose=1)
    ec_model.learn(total_timesteps=budget)


if __name__ == '__main__':
    ec_cartpole_example()
