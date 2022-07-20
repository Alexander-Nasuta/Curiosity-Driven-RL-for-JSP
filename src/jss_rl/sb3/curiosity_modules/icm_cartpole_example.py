from jss_rl.sb3.curiosity_modules.icm_wrapper import IntrinsicCuriosityModuleWrapper
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO


def icm_cartpole_example(total_timesteps:int = 10_000) -> None:
    env_id = "CartPole-v1"

    venv = make_vec_env_without_monitor(
        env_id=env_id,
        env_kwargs={},
        n_envs=4
    )
    cartpole_venv = VecMonitor(venv=venv)

    cartpole_venv.reset()
    cartpole_icm_venv = IntrinsicCuriosityModuleWrapper(
        venv=venv,
    )

    icm_model = PPO('MlpPolicy', cartpole_icm_venv, verbose=1)
    icm_model.learn(total_timesteps=total_timesteps)


if __name__ == '__main__':
    icm_cartpole_esxample()