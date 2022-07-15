import pprint

import gym
import gym_minigrid

import numpy as np

from ray import tune
from tabulate import tabulate
from collections import deque

from jss_rl.rllib.icm2 import Curiosity3
from jss_utils.jss_logger import log

from ray.rllib.agents import DefaultCallbacks
from ray.rllib.agents.ppo import ppo
from ray.rllib.utils.numpy import one_hot
from ray.tune import register_env
from ray.rllib.utils.test_utils import check_learning_achieved, framework_iterator


class MyCallBack(DefaultCallbacks):
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


class OneHotWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, vector_index, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.single_frame_dim = 49 * (11 + 6 + 3) + 4
        self.init_x = None
        self.init_y = None
        self.x_positions = []
        self.y_positions = []
        self.x_y_delta_buffer = deque(maxlen=100)
        self.vector_index = vector_index
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.single_frame_dim,)))

        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(self.single_frame_dim * self.framestack,), dtype=np.float32
        )

    def observation(self, obs):
        # Debug output: max-x/y positions to watch exploration progress.
        if self.step_count == 0:
            for _ in range(self.framestack):
                self.frame_buffer.append(np.zeros((self.single_frame_dim,)))
            if self.vector_index == 0:
                if self.x_positions:
                    max_diff = max(
                        np.sqrt(
                            (np.array(self.x_positions) - self.init_x) ** 2
                            + (np.array(self.y_positions) - self.init_y) ** 2
                        )
                    )
                    self.x_y_delta_buffer.append(max_diff)
                    print(
                        f"100-average dist travelled={np.mean(self.x_y_delta_buffer):.4}"
                    )
                    self.x_positions = []
                    self.y_positions = []
                self.init_x = self.agent_pos[0]
                self.init_y = self.agent_pos[1]

        # Are we carrying the key?
        # if self.carrying is not None:
        #    print("Carrying KEY!!")

        self.x_positions.append(self.agent_pos[0])
        self.y_positions.append(self.agent_pos[1])

        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(obs[:, :, 0], depth=11)
        colors = one_hot(obs[:, :, 1], depth=6)
        states = one_hot(obs[:, :, 2], depth=3)
        # Is the door we see open?
        # for x in range(7):
        #    for y in range(7):
        #        if objects[x, y, 4] == 1.0 and states[x, y, 0] == 1.0:
        #            print("Door OPEN!!")

        all_ = np.concatenate([objects, colors, states], -1)
        all_flat = np.reshape(all_, (-1,))
        direction = one_hot(np.array(self.agent_dir), depth=4).astype(np.float32)
        single_frame = np.concatenate([all_flat, direction])
        self.frame_buffer.append(single_frame)
        return np.concatenate(self.frame_buffer)


def env_maker(config):
    name = config.get("name", "MiniGrid-Empty-5x5-v0")
    framestack = config.get("framestack", 4)
    env = gym.make(name)
    # Only use image portion of observation (discard goal and direction).
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    env = OneHotWrapper(
        env,
        config.vector_index if hasattr(config, "vector_index") else 0,
        framestack=framestack,
    )
    return env


register_env("mini-grid", env_maker)


def test_curiosity_on_frozen_lake():
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
    config["callbacks"] = MyCallBack
    # Limit horizon to make it really hard for non-curious agent to reach
    # the goal state.
    config["horizon"] = 16
    # Local only.
    config["num_workers"] = 0
    config["lr"] = 0.001
    config["framework"] = "torch"

    no_icm_config = config.copy()
    trainer_without_icm = ppo.PPOTrainer(config=no_icm_config)
    config["exploration_config"] = {
        "type": Curiosity3,  # <- Use the Curiosity module for exploring.
        "framework": "torch",
        "eta": 1.0,  # Weight for intrinsic rewards before being added to extrinsic ones.
        "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
        "feature_dim": 288,  # Dimensionality of the generated feature vectors.
        # Setup of the feature net (used to encode observations into feature (latent) vectors).
        "feature_net_config": {
            "fcnet_hiddens": [],
            "fcnet_activation": "relu",
        },
        "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
        "inverse_net_activation": "relu",  # Activation of the "inverse" model.
        "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
        "forward_net_activation": "relu",  # Activation of the "forward" model.
        "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
        # Specify, which exploration sub-type to use (usually, the algo's "default"
        # exploration, e.g. EpsilonGreedy for DQN, StochasticSampling for PG/SAC).
        "sub_exploration": {
            "type": "StochasticSampling",
        }
    }
    icm_config = config.copy()
    print(pprint.pformat(icm_config))
    trainer_with_icm = ppo.PPOTrainer(config=icm_config)

    headers = ['setup', 'reached goal', 'iterations']
    table_data = [
        ['ppo with icm', False, '-'],
        ['ppo without icm', False, '-']
    ]

    num_iterations = 10
    learnt = False
    log.info("evaluating performance with icm")
    for i in range(num_iterations):
        result = trainer_with_icm.train()
        log.info(pprint.pformat(result))
        if result["episode_reward_max"] > 0.0:
            log.info(f"reached goal after {i} iters!")
            learnt = True
            table_data[0][1] = learnt
            table_data[0][2] = i
            break
    trainer_with_icm.stop()

    learnt = False
    log.info("evaluating performance without icm")
    for i in range(0):
        result = trainer_without_icm.train()
        log.info(pprint.pformat(result))
        if result["episode_reward_max"] > 0.0:
            log.info(f"reached goal after {i} iters!")
            learnt = True
            table_data[1][1] = learnt
            table_data[1][2] = i
            break

    log.info(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    log.info(f"the iteration limit was {num_iterations}")


def test_curiosity_on_partially_observable_domain():
    log.info("comparing icm performance with ppo algorithm on 'MiniGrid' enviorment")
    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = "mini-grid"
    config["env_config"] = {
        # Also works with:
        # - MiniGrid-MultiRoom-N4-S5-v0
        # - MiniGrid-MultiRoom-N2-S4-v0
        "name": "MiniGrid-Empty-8x8-v0",
        "framestack": 1,  # seems to work even w/o framestacking
    }
    config["horizon"] = 15  # Make it impossible to reach goal by chance.
    config["num_envs_per_worker"] = 4
    config["model"]["fcnet_hiddens"] = [256, 256]
    config["model"]["fcnet_activation"] = "relu"
    config["num_sgd_iter"] = 8
    config["num_workers"] = 0

    no_icm_config = config.copy()

    config["exploration_config"] = {
        "type": Curiosity3,
        "framework": "torch",
        # For the feature NN, use a non-LSTM fcnet (same as the one
        # in the policy model).
        "eta": 0.1,
        "lr": 0.0003,  # 0.0003 or 0.0005 seem to work fine as well.
        "feature_dim": 64,
        # No actual feature net: map directly from observations to feature
        # vector (linearly).
        "feature_net_config": {
            "fcnet_hiddens": [],
            "fcnet_activation": "relu",
        },
        "sub_exploration": {
            "type": "StochasticSampling",
        },
    }

    icm_config = config.copy()

    headers = ['setup', 'reached goal', 'iterations']
    table_data = [
        ['ppo with icm', False, '-'],
        ['ppo without icm', False, '-']
    ]

    min_reward = 0.001
    stop = {
        "training_iteration": 50,
        "episode_reward_mean": min_reward,
    }
    for _ in framework_iterator(icm_config, frameworks="torch"):
        results = tune.run("PPO", config=icm_config, stop=stop, verbose=1)
        try:
            check_learning_achieved(results, min_reward)
            iters = results.trials[0].last_result["training_iteration"]
            log.info(f"reached in {iters} iterations.")
            table_data[0][1] = True
            table_data[0][2] = iters
        except ValueError:  # catch `stop-reward` of <...> not reached!` from check_learning_achieved
            table_data[0][1] = False

    for _ in framework_iterator(no_icm_config, frameworks="torch"):
        results = tune.run("PPO", config=no_icm_config, stop=stop, verbose=1)
        try:
            check_learning_achieved(results, min_reward)
            iters = results.trials[0].last_result["training_iteration"]
            log.info(f"reached in {iters} iterations.")
            table_data[1][1] = True
            table_data[1][2] = iters
        except ValueError: # catch `stop-reward` of <...> not reached!` from check_learning_achieved
            table_data[1][1] = False

    log.info(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    log.info(f"the training_iteration limit was {stop['training_iteration']}")


if __name__ == '__main__':
    test_curiosity_on_frozen_lake()
    #test_curiosity_on_partially_observable_domain()
