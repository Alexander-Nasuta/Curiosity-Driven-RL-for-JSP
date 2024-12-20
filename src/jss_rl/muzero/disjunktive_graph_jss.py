import datetime
import pathlib

import numpy as np
import torch
from muzero_baseline import AbstractGame, MuZero

from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here:
        # https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it
        # has enough memory. None will use every GPUs available
        self.max_num_gpus = None

        # Game
        # Dimensions of the game observation, must be 3D (channel, height, width).
        # For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = (1, 1,
                                  1548)
        self.action_space = list(range(36))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        # Number of previous observations and previous actions to add to the current observation
        self.stacked_observations = 0

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        # Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        # It doesn't influence training. None, "random" or "expert" if implemented in the Game class
        self.opponent = None

        # Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 36  # Maximum number of moves if game is not finished before
        self.num_simulations = 20  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0
        # (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "fullyconnected"  # "resnet" / "fullyconnected"
        # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of
        # -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = 10

        # Residual Network
        # Downsample observations before representation network, False / "CNN" (lighter) / "resnet"
        # (See paper appendix Network Architecture)
        self.downsample = False
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 2  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = []  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = []  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = []  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 8
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network

        # Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(
            __file__).stem / datetime.datetime.now().strftime(
            "%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 20_000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 10  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25
        # (See paper appendix Reanalyze)
        self.value_loss_weight = 1
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.02  # Initial learning rate
        self.lr_decay_rate = 0.8  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 1000

        # Replay Buffer
        self.replay_buffer_size = 500  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        # Prioritized Replay (See paper appendix Training),
        # select in priority the elements in the replay buffer which are unexpected for the network
        self.PER = True
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        # self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        # Desired training steps per self played step ratio. Equivalent to a synchronous version,
        # training can take much longer. Set it to None to disable it
        self.ratio = 1.5
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None, env_kwargs: dict = {}):
        print("init")
        self.env = DisjunctiveGraphJssEnv(**env_kwargs)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        # print(f"{action=}")
        observation, reward, done, _ = self.env.step(action)
        return np.array([[observation]]), reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        legal_actions = list([i for i, bol in enumerate(self.env.valid_action_mask()) if bol])
        # print(f"{legal_actions=}")
        return legal_actions

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        # print("reset")
        return np.array([[self.env.reset()]])

    def close(self):
        """
        Properly close the game.
        """
        print("close")
        self.env.close()

    def render(self, **kwargs):
        """
        Display the game observation.
        :param **kwargs:
        """
        print("render")
        self.env.render(**kwargs)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            i: f"Task {i} (Job {i // self.env.n_machines}, duration {self.env.G.nodes[i]['duration']})"
            for i in range(self.env.total_tasks_without_dummies)}
        print(f"{actions=}")
        return f"{action_number}. {actions[action_number]}"


if __name__ == '__main__':
    instance_name = "ft06"
    jsp = parser.get_instance_by_name(instance_name)
    lb = details.get_jps_instance_details(instance_name)["lower_bound"]

    # Initialize config
    config = MuZeroConfig()
    # Game object will be initialized in each thread separetly
    mz = MuZero(Game, config,
                game_kwargs={
                    "env_kwargs": {
                        "jps_instance": jsp,
                        "scaling_divisor": lb,
                        "action_mode": "task"
                    }
                })
    mz.train()
