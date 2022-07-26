from jss_rl.sb3.curiosity_modules.ec_cartpole_example import ec_cartpole_example
from jss_rl.sb3.curiosity_modules.ec_frozenlake_example import ec_frozenlake_example


def test_ec_on_cartpole():
    ec_cartpole_example(total_timesteps=1_000, n_envs=4)


def test_ec_on_frozenlake():
    ec_frozenlake_example(total_timesteps=1_000, n_envs=1)
