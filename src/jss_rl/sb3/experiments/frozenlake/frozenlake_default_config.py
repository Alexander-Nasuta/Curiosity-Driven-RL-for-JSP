import torch as th
from gym.wrappers import TimeLimit

frozenlake_default_config = {
    "env_name": "FrozenLake-v1",
    "env_kwargs": {
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
    },
    "wrapper_class": TimeLimit,
    "wrapper_kwargs": {"max_episode_steps": 16},
    "n_envs": 8,
    "model_policy": "MlpPolicy",
    "model_hyper_parameters": {
        "gamma": 0.99,  # discount factor,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "clip_range": 0.541,
        "clip_range_vf": 26,
        "ent_coef": 0.0,
        "normalize_advantage": True,
        # "target_kl": 0.005047, # specify some value for early stopping
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
    "curiosity_module": "None"
}