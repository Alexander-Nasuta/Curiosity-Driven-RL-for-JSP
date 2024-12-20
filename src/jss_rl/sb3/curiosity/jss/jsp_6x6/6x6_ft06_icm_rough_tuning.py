import gym

import wandb as wb

import jss_utils.PATHS as PATHS

wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))

gym.envs.register(
    id='GraphJsp-v0',
    entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
    kwargs={},
)



SWEEP_CONFIG = {
    'method': 'random',
    'metric': {
        'name': 'mean_makespan',
        'goal': 'minimize'
    },
    'parameters': {
        "trigger": {
            "values": ["step", "episode"]
        },
        # only relevant for trigger = 'step'
        "clear_memory_every_n_steps": {
            'values': [
                36*2,
                36*5,
                1_000,
                36*10,
                5_000,
                10_000
            ]
        },
        # only relevant for trigger = 'step'
        "postprocess_every_n_steps": {
            'values': [
                6,
                9,
                18,
                36*1,
                36*5,
            ]
        },
        "beta": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },
        "eta": {
            'values': [
                1.0,
                0.1,
                0.01,
                0.001,
                0.000_1,
                0.000_01,
                0.000_001,
                0.000_000_1
            ]
        },
        "memory_capacity": {
            'values': [
                36,
                36*2,
                36*5,
                1_000,
                5_000,
                10_000
            ]
        },
        # % of memory_capacity
        "maximum_sample_size_pct": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 1.0,
        },
        "lr":  {
            'values': [
                1e-2,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
            ]
        },
        "feature_dim":  {
            'values': [
                288 * 0.5,
                288,
                288 * 2,
                288 * 3,
                288 * 4,
                288 * 5,
                288 * 10,
            ]
        },

        "feature_net_n_layers": {
            'values': [1, 2, 3, 4, 5]
        },
        "feature_net_layer_size": {
            'values': [8, 16, 32, 64, 128, 256, 512, 1024]
        },

        "inverse_feature_net_n_layers": {
            'values': [1, 2, 3, 4, 5]
        },
        "inverse_feature_net_layer_size": {
            'values': [8, 16, 32, 64, 128, 256, 512, 1024]
        },

        "forward_fcnet_net_n_layers": {
            'values': [1, 2, 3, 4, 5]
        },
        "forward_fcnet_net_layer_size": {
            'values': [8, 16, 32, 64, 128, 256, 512, 1024]
        },

        "exploration_steps": {
            # total steps: 100_000
            'values': [
                None,
                25_000,
                50_000,
                75_000,
            ]
        },
        "shuffle_samples":  {
            'values': [
                True, False
            ]
        },


        # eval params
        "n_eval_episodes": {
            'value': 50
        }
    }

}