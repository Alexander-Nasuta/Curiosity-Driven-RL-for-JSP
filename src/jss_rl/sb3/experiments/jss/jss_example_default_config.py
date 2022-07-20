import torch as th
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy


# insert parameters from hyper-param tuning here
jss_default_config = {
    "n_envs": 8,
    "policy_type": MaskableActorCriticPolicy,
    "model_hyper_parameters": {
        "gamma": 0.99999,  # discount factor,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "clip_range": 0.541,
        "clip_range_vf": None,
        "ent_coef": 0.0,
        "normalize_advantage": True,
        # "target_kl": 0.005047, # for early stopping
        "policy_kwargs": {
            "net_arch": [{
                "pi": [64, 64],
                "vf": [64, 64],
            }],
            "ortho_init": True,
            "activation_fn": th.nn.Tanh,  # th.nn.ReLU
            "optimizer_kwargs": {  # for th.optim.Adam
                "eps": 1e-5
            }
        }
    },
    "env_name": "GraphJsp-v0",
    "env_kwargs": {
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True,
        "perform_left_shift_if_possible": True,
        "default_visualisations": [
            "gantt_window",
            # "graph_window",  # very expensive
            "gantt_console",
            "graph_console",
        ]
    },
    # wb run config
    "sync_tensorboard": True,
    "monitor_gym": True,
    "save_code": True,
}
