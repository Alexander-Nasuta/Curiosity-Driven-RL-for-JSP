import pprint

import gym
import gym_minigrid

import numpy as np

from ray import tune
from collections import deque

from jss_rl.rllib.ec import EpisodicCuriosity
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
    log.info("episodic curiosity with ppo algorithm on 'FrozenLake-v1' enviorment")
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

    print(pprint.pformat(config))
    trainer_with_icm = ppo.PPOTrainer(config=config)

    num_iterations = 10
    learnt = False
    log.info("evaluating performance with icm")
    for i in range(num_iterations):
        result = trainer_with_icm.train()
        # log.info(pprint.pformat(result))
        if result["episode_reward_max"] > 0.0:
            log.info(f"reached goal after {i} iters!")
            learnt = True
            break
    trainer_with_icm.stop()

    log.info(f"the iteration limit was {num_iterations}. Goal reached: {learnt}")


def test_curiosity_on_partially_observable_domain():
    log.info("episodic curiosity with ppo algorithm on 'mini-grid' enviorment")
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

    config["exploration_config"] = {
        "type": EpisodicCuriosity,
        "framework": "torch",
        # For the feature NN, use a non-LSTM fcnet (same as the one
        # in the policy model).
        "alpha": 0.1,
        "beta": 1.0,
        "lr": 0.0005,  # 0.0003 or 0.0005 seem to work fine as well.
        "embedding_dim": 64,
        "episodic_memory_capacity": 50,
        "embedding_net_config": {
            "fcnet_hiddens": [],
            "fcnet_activation": "relu",
        },
        "sub_exploration": {
            "type": "StochasticSampling",
        },
    }

    min_reward = 0.001
    stop = {
        "training_iteration": 25,
        "episode_reward_mean": min_reward,
    }
    for _ in framework_iterator(config, frameworks="torch"):
        results = tune.run("PPO", config=config, stop=stop, verbose=1)
        try:
            check_learning_achieved(results, min_reward)
            iters = results.trials[0].last_result["training_iteration"]
            log.info(f"reached in {iters} iterations.")
        except ValueError:  # catch `stop-reward` of <...> not reached!` from check_learning_achieved
            log.info(f"reward not reached.")

    log.info(f"the training_iteration limit was {stop['training_iteration']}")


if __name__ == '__main__':
    test_curiosity_on_frozen_lake()
    # test_curiosity_on_partially_observable_domain()
