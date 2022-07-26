from jss_rl.sb3.curiosity_modules.icm_wrapper import IntrinsicCuriosityModuleWrapper
from jss_rl.sb3.make_vec_env_without_monitor import make_vec_env_without_monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO


def icm_cartpole_example(total_timesteps: int = 10_000) -> None:
    env_id = "CartPole-v1"

    venv = make_vec_env_without_monitor(
        env_id=env_id,
        env_kwargs={},
        n_envs=4
    )

    venv.reset()

    kwargs = {'beta': 0.9299789286150444,
              'eta': 0.008175724180340696,
              'exploration_steps': 60000,
              'feature_dim': 1584,
              'feature_net_activation': 'relu',
              'feature_net_hiddens': [90, 90, 90],
              'forward_fcnet_net_activation': 'relu',
              'forward_fcnet_net_hiddens': [80, 80],
              'inverse_feature_net_activation': 'relu',
              'inverse_feature_net_hiddens': [90, 90, 90],
              'lr': 1.894747356782694e-05,
              'maximum_sample_size': 213,
              'memory_capacity': 480,
              'shuffle_samples': True,

              'postprocess_on_end_of_episode': (True,),
              'clear_memory_on_end_of_episode': True,

              'postprocess_every_n_steps': 360,
              'clear_memory_every_n_steps': 220,
              }
    venv = IntrinsicCuriosityModuleWrapper(
        venv=venv,
    )
    venv = VecMonitor(venv=venv)

    icm_model = PPO('MlpPolicy', venv, verbose=1)
    icm_model.learn(total_timesteps=total_timesteps)


if __name__ == '__main__':
    icm_cartpole_example()
