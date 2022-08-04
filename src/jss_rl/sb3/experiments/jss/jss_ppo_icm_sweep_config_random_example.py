# non dynamic instance params from sweeps
from jss_utils.name_generator import generate_name

tuning_results = {
        # Constanst
        "curiosity_module": {
            'values': [None]
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

        # gamma: float = 0.99,
        # Discount factor
        "gamma": {
            "values": [0.99013]
        },
        # gae_lambda: float = 0.95,
        # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        "gae_lambda": {
            "values": [0.9, 0.95]
        },
        # max_grad_norm: float = 0.5,
        # The maximum value for the gradient clipping
        "max_grad_norm": {
            "values": [0.5]
        },

        # learning_rate: Union[float, Schedule] = 3e-4,
        # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
        "learning_rate": {
            "values": [6e-4, 2e-4]
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
            'values': [1, 2]
        },
        "net_arch_n_size": {
            "values": [90, 20]
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
                "ReLu",  # th.nn.ReLU
                "ELU",
                "RRELU",
            ]
        },

        "optimizer_eps": {  # for th.optim.Adam
            "values": [1e-7, 1e-8]
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
            'values': [True, False]
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
    'name': generate_name(),
    'metric': {
        'name': 'mean_makespan',
        'goal': 'minimize'
    },
    'parameters': {
        **tuning_results,

        # Constants
        "curiosity_module": {
            'values': ["icm"]
        },
        "total_timesteps": {
            'values': [50_000]
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

        # icm params
        "beta": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },

        "eta": {
            'distribution': 'log_uniform_values',
            'min': 1e-6,
            'max': 1e0,
        },


        "feature_dim": {
            'distribution': 'q_log_uniform_values',
            'max': 3e3,
            'min': 1e2,
            "q": 72
        },

        "lr": {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-3,
        },

        "feature_net_activation": {
            "values": ["relu"]
        },
        "inverse_feature_net_activation": {
            "values": ["relu"]
        },
        "forward_fcnet_net_activation": {
            "values": ["relu"]
        },
        "memory_capacity": {
            'distribution': 'q_log_uniform_values',
            'min': 36e1,
            'max': 36e3,
            "q": 36
        },

        "exploration_steps": {
            "values": [
                None,
                35_000,
                30_000,
                25_000,
                20_000
            ]
        },

        "maximum_sample_size_pct": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        },

        "feature_nets_n_layers": {
            'values': [1, 2, 3]
        },
        "feature_nets_arch_n_size": {
            "distribution": "q_uniform",
            "min": 20,
            "max": 100,
            "q": 20
        },

        "forward_net_n_layers": {
            'values': [1, 2, 3]
        },
        "forward_net_arch_n_size": {
            "distribution": "q_uniform",
            "min": 20,
            "max": 100,
            "q": 20
        },

        "icm_postprocess_trigger": {
            "values": [
                'step',
                'episode'
            ]
        },

        "icm_memory_clearing": {
            "values": [True, False]
        },

        "clear_memory_every_n_steps": {
            'distribution': 'q_log_uniform_values',
            'max': 36e5,
            'min': 72e2,
            "q": 360
        },

        "postprocess_every_n_steps": {
            'distribution': 'q_log_uniform_values',
            'max': 36e4,
            'min': 36e2,
            "q": 72
        },

    }
}


if __name__ == '__main__':
    import wandb as wb
    from jss_rl.sb3.experiments.jss.jss_ppo_perform_sweep_run import perform_jss_run
    sweep_id = wb.sweep(ppo_icm_sweep_config_random, project="testo")
    # sweep_id = "sen6u09f"  # icm random sweep
    wb.agent(
        sweep_id,
        function=perform_jss_run,
        count=3,
        project="testo"
    )