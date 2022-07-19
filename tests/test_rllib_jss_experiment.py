from jss_rl.rllib.jss_experiment import run_experiment


def test_icm_ppo_tune(abz5):
    jsp_instance, jsp_instance_details = abz5
    stop = {
        "training_iteration": 1,
        "episode_reward_mean": -10.00,
    }

    run_experiment(
        jsp_instance=jsp_instance,
        jsp_instance_details=jsp_instance_details,
        stop_conditions=stop,
        num_samples=1,
        wandb_project="testing"
    )