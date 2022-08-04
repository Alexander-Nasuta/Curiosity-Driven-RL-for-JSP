from pathlib import Path

import wandb as wb
import jss_utils.PATHS as PATHS
from jss_rl.sb3.experiments.jss.cost_estimator import extrapolate_cost

from jss_rl.sb3.experiments.jss.jss_ppo_perform_sweep_run import perform_jss_run
from argparse import ArgumentParser
from jss_utils.name_generator import generate_name
from jss_utils.jss_logger import log

tuning_results = {
        # gamma: float = 0.99,
        # Discount factor
        "gamma": {
            "values": [0.99013]
        },
        # gae_lambda: float = 0.95,
        # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        "gae_lambda": {
            "values": [0.9]
        },
        # max_grad_norm: float = 0.5,
        # The maximum value for the gradient clipping
        "max_grad_norm": {
            "values": [0.5]
        },

        # learning_rate: Union[float, Schedule] = 3e-4,
        # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
        "learning_rate": {
            "values": [6e-4]
        },

        # batch_size: Optional[int] = 64,
        # Minibatch size
        "batch_size": {
            "values": [64]
        },
        # clip_range: Union[float, Schedule] = 0.2,
        # Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
        "clip_range": {
            'values': [0.2]
        },

        # clip_range_vf: Union[None, float, Schedule] = None,
        #
        # Clipping parameter for the value function,
        # it can be a function of the current progress remaining (from 1 to 0).
        # This is a parameter specific to the OpenAI implementation.
        # If None is passed (default), no clipping will be done on the value function.
        #
        # IMPORTANT: this clipping depends on the reward scaling.
        #
        "clip_range_vf": {
            'values': [
                None
            ]
        },

        # vf_coef: float = 0.5,
        # Value function coefficient for the loss calculation
        "vf_coef": {
            'values': [0.5]
        },

        # ent_coef: float = 0.0,
        # Entropy coefficient for the loss calculation
        "ent_coef": {
            'values': [0.0]
        },

        # normalize_advantage: bool = True
        # Whether to normalize or not the advantage
        "normalize_advantage": {
            'values': [True]
        },
        # n_epochs: int = 10,
        # Number of epoch when optimizing the surrogate loss
        "n_epochs": {
            'values': [28]
        },

        # n_steps: int = 2048,
        # The number of steps to run for each environment per update
        # (i.e. batch size is n_steps * n_env where n_env is number of environment
        # copies running in parallel)
        "n_steps": {
            'values': [432]
        },
        # device: Union[th.device, str] = "auto",
        #  Device (cpu, cuda, …) on which the code should be run. Setting it to auto,
        #  the code will be run on the GPU if possible.
        "device": {
            "values": ["auto"]
        },
        # seed: Optional[int] = None,
        # Seed for the pseudo random generators
        "seed": {
            "values": [None]
        },

        # verbose: int = 0,
        # the verbosity level: 0 no output, 1 info, 2 debug
        # "verbose": {
        #     "values": [0]
        # },

        # create_eval_env: bool = False,
        # Whether to create a second environment that will be used for evaluating the agent periodically.
        # (Only available when passing string for the environment)
        "create_eval_env": {
            "values": [False]
        },
        # tensorboard_log: Optional[str] = None,
        # the log location for tensorboard (if None, no logging)
        # "tensorboard_log": {
        #    "values": [None]
        # },

        # target_kl: Optional[float] = None,
        # Limit the KL divergence between updates, because the clipping
        # is not enough to prevent large update
        # see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        # By default, there is no limit on the kl div.
        "target_kl": {
            "values": [None]
        },

        # policy params

        # net_arch (Optional[List[Union[int, Dict[str, List[int]]]]]) –
        # The specification of the policy and value networks.

        # 'net_arch_n_layers' and 'net_arch_n_size' will result in a dict that will be passed to 'net_arch'
        # see code below
        "net_arch_n_layers": {
            'values': [2]
        },
        "net_arch_n_size": {
            "values": [90]
        },

        # ortho_init: bool = True,
        # Whether to use or not orthogonal initialization
        "ortho_init": {
            'values': [True]
        },
        # normalize_images: bool = True,
        "normalize_images": {
            'values': [True]
        },
        # activation_fn: Type[nn.Module] = nn.Tanh
        # Activation function
        # https://pytorch.org/docs/stable/nn.html
        "activation_fn": {
            "values": [
                "RRELU",
            ]
        },

        "optimizer_eps": {  # for th.optim.Adam
            "values": [1e-7]
        },

        # env params
        "action_mode": {
            'values': ['task']
        },
        "normalize_observation_space": {
            'values': [True]
        },
        "flat_observation_space": {
            'values': [True]
        },
        "perform_left_shift_if_possible": {
            'values': [True]
        },
        "dtype": {
            'values': ["float32"]
        },

        # eval params
        "n_eval_episodes": {
            'value': 50
        }
    }


ppo_icm_sweep_config_random = {
    'method': 'random',
    'name': f"{generate_name()}_{Path(__file__).stem}",
    'metric': {
        'name': 'mean_makespan',
        'goal': 'minimize'
    },
    'parameters': {
        **tuning_results,

        # Constants
        "curiosity_module": {
            'values': ["ec"]
        },
        "total_timesteps": {
            'values': [40_000]
        },
        "n_envs": {
            'values': [8]
        },
        "benchmark_instance": {
            'values': ["ft06"]
        },
        "dynamic_instances": {
            'values': [False]
        },
        "n_machines": {
            "values": [6]
        },
        "n_jobs": {
            "values": [6]
        },

        # ec params
        "alpha": {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e-3,
        },

        "beta": {
            'distribution': 'q_uniform',
            'min': 0.4,
            'max': 1.0,
            'q': 0.2
        },

        "b_novelty": {
            'distribution': 'q_uniform',
            'min': 0.0,
            'max': 0.5,
            'q': 0.15,
        },

        "ec_gamma": {
            "values": [2, 3]
        },

        "k": {
            "values": [2, 3]
        },

        "lr": {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2,
        },

        "episodic_memory_capacity": {
            'distribution': 'q_log_uniform_values',
            'max': 1e4,
            'min': 1e2,
            "q": 100
        },

        "clear_memory_every_episode": {
            "values": [False]
        },

        "embedding_net_activation": {
            "values": ["relu"]
        },

        "embedding_net_n_layers": {
            'values': [1, 2, 3]
        },
        "embedding_net_arch_n_size": {
            "distribution": "q_uniform",
            "min": 25,
            "max": 100,
            "q": 25
        },

        "comparator_net_activation": {
            "values": ["relu"]
        },

        "comparator_net_n_layers": {
            'values': [1, 2, 3]
        },
        "comparator_net_arch_n_size": {
            "distribution": "q_uniform",
            "min": 25,
            "max": 100,
            "q": 25
        },

        "exploration_steps": {
            "values": [
                None,
                30_000,
            ]
        },

    }
}

def run_sweep(project: str, sweep_id: str = None, new_sweep: bool = False, count: int = None, **_) -> None:
    wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))
    if not project:
        raise ValueError("'project' must not be `None`!")
    if sweep_id and new_sweep:
        raise ValueError("'sweep_id' must be `None` if 'new_sweep' is `True`")
    if new_sweep:
        sweep_id = wb.sweep(ppo_icm_sweep_config_random, project=project)
        if count is None:
            log.info(f"your 'sweep_id' is `{sweep_id}`. rerun this script with the '-c'/'--count'-argument specified "
                     f"to perform runs for the sweep.")
            return
    if count is None:
        raise ValueError("for sweeps with `method='random'` 'count' must be specified!")
    wb.agent(
        sweep_id,
        function=perform_jss_run,
        count=count,
        project=project
    )


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-p", "--project",
                        dest="project",
                        help="the wandb project to which the result will be logged",
                        default="MA-nasuta"
                        )

    parser.add_argument("-sid", "--sweep_id",
                        dest="sweep_id",
                        help="the id of an existing sweep in wandb. "
                             "The performed runs will be logged to the specified sweep.",
                        default=None
                        )

    parser.add_argument("-n", "--new",
                        dest="new_sweep",
                        type=bool,
                        help="a new sweep will be created if set to `True`",
                        default=False
                        )

    parser.add_argument("-c", "--count",
                        dest="count",
                        type=int,
                        help="the number of runs that shall be performed",
                        default=None
                        )

    parser.add_argument("-cost", "--cost",
                        dest="cost",
                        type=bool,
                        help="extrapolate cost based on one run",
                        default=False
                        )

    args = vars(parser.parse_args())
    if args["cost"]:
        extrapolate_cost(
            callable_function=run_sweep,
            function_kwargs={**args, "count": 1}
        )
    else:
        run_sweep(**args)
