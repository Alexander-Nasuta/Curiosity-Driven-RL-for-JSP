from jss_rl.rllib.jss_basic_ppo import run_ppo_with_tune, run_ppo_with_trainer
from jss_rl.rllib.jss_ec_ppo import run_ec_ppo_with_tune, run_ec_ppo_with_trainer
from jss_rl.rllib.jss_icm_ppo import run_icm_ppo_with_tune, run_icm_ppo_with_trainer
from jss_rl.rllib.jss_mask_ppo import run_mask_ppo_with_tune, run_mask_ppo_trainer


def test_basic_ppo_tune(abz5):
    jsp_instance, jsp_instance_details = abz5
    run_ppo_with_tune(jsp_instance, jsp_instance_details)


def test_basic_ppo_trainer(abz5):
    jsp_instance, jsp_instance_details = abz5
    run_ppo_with_trainer(jsp_instance, jsp_instance_details)


def test_mask_ppo_tune(abz5):
    jsp_instance, jsp_instance_details = abz5
    run_mask_ppo_with_tune(jsp_instance, jsp_instance_details)


def test_mask_ppo_trainer(abz5):
    jsp_instance, jsp_instance_details = abz5
    run_mask_ppo_trainer(jsp_instance, jsp_instance_details)


def test_icm_ppo_tune(abz5):
    jsp_instance, jsp_instance_details = abz5
    run_icm_ppo_with_tune(jsp_instance, jsp_instance_details)


def test_icm_ppo_trainer(abz5):
    jsp_instance, jsp_instance_details = abz5
    run_icm_ppo_with_trainer(jsp_instance, jsp_instance_details)

def test_ec_ppo_tune(abz5):
    jsp_instance, jsp_instance_details = abz5
    run_ec_ppo_with_tune(jsp_instance, jsp_instance_details)


def test_ec_ppo_trainer(abz5):
    jsp_instance, jsp_instance_details = abz5
    run_ec_ppo_with_trainer(jsp_instance, jsp_instance_details)
