from jss_rl.rllib.frozenlake_ec_ppo import run_ec_on_frozenlake
from jss_rl.rllib.frozenlake_icm_ppo import run_icm_on_frozenlake


def test_icm_frozenlake():
    run_icm_on_frozenlake()


def test_ec_frozenlake():
    run_ec_on_frozenlake()
