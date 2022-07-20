from jss_rl.sb3.curiosity_modules.icm_cartpole_example import icm_cartpole_example
from jss_rl.sb3.curiosity_modules.icm_frozenlake_example import icm_frozenlake_example


def test_icm_on_cartpole():
    icm_cartpole_example(total_timesteps=1_000)


def test_icm_on_frozenlake():
    icm_frozenlake_example(total_timesteps=1_000)
