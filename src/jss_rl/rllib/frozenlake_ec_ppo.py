import numpy as np

from ray.rllib.agents import DefaultCallbacks
from ray.rllib.agents.ppo import ppo

from jss_rl.rllib.episodic_curiosity_module import EpisodicCuriosity
from jss_utils.jss_logger import log


class FrozenlakeCallBack(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.deltas = []

    def on_postprocess_trajectory(
            self,
            *,
            worker,
            episode,
            agent_id,
            policy_id,
            policies,
            postprocessed_batch,
            original_batches,
            **kwargs
    ):
        pos = np.argmax(postprocessed_batch["obs"], -1)
        x, y = pos % 8, pos // 8
        self.deltas.extend((x ** 2 + y ** 2) ** 0.5)

    def on_sample_end(self, *, worker, samples, **kwargs):
        print(f"mean. distance from origin={np.mean(self.deltas):.4}")
        self.deltas = []


def run_ec_on_frozenlake():
    log.info("comparing icm performance with ppo algorithm on 'FrozenLake-v1' enviorment")
    config = ppo.DEFAULT_CONFIG.copy()
    # A very large frozen-lake that's hard for a random policy to solve
    # due to 0.0 feedback.
    config["env"] = "FrozenLake-v1"
    config["env_config"] = {
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
    # Print out observations to see how far we already get inside the Env.
    config["callbacks"] = FrozenlakeCallBack
    # Limit horizon to make it really hard for non-curious agent to reach
    # the goal state.
    config["horizon"] = 16
    # Local only.
    config["num_workers"] = 0
    config["lr"] = 0.001
    config["framework"] = "torch"

    config["exploration_config"] = {
        "type": EpisodicCuriosity,
        "framework": "torch",
        # For the feature NN, use a non-LSTM fcnet (same as the one
        # in the policy model).
        "alpha": 1.0,
        "beta": 1.0,
        "embedding_dim": 64,
        "episodic_memory_capacity": 8,

        "sub_exploration": {
            "type": "StochasticSampling",
        },
    }
    trainer_with_icm = ppo.PPOTrainer(config=config)

    num_iterations = 10
    learnt = False
    for i in range(num_iterations):
        result = trainer_with_icm.train()
        if result["episode_reward_max"] > 0.0:
            log.info(f"reached goal after {i} iters!")
            learnt = True
            break
    trainer_with_icm.stop()

    assert learnt


if __name__ == '__main__':
    run_ec_on_frozenlake()
