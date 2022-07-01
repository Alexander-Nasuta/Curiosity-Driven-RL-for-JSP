import math
import pprint

import gym
import numpy as np
import sb3_contrib
import torch as th
import wandb as wb

import jss_utils.PATHS as PATHS
import jss_utils.jsp_env_utils as env_utils

from types import ModuleType
from rich.progress import track

from wandb.integration.sb3 import WandbCallback

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder

from jss_rl.sb3.curiosity.curiosity_info_wrapper import CuriosityInfoWrapper
from jss_rl.sb3.curiosity.icm2 import IntrinsicCuriosityModuleWrapper
from jss_rl.sb3.curiosity.jss.jss_logger_callback import JssLoggerCallback
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_utils.jss_logger import log

wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))

gym.envs.register(
    id='GraphJsp-v0',
    entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
    kwargs={},
)

PROJECT = "6x6_ft06_tuning"

SWEEP_CONFIG = {
    'method': 'random',
    'metric': {
        'name': 'mean_makespan',
        'goal': 'minimize'
    },
    "name": "6x6_ft06_mask_ppo_fine_tuning_random2",

    'parameters': {
        # gamma: float = 0.99,
        # Discount factor
        "gamma": {
            'values': [
                1.0,
                0.99,
                0.9
            ]
        },
        # gae_lambda: float = 0.95,
        # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        "gae_lambda": {
            'values': [
                1.0,
                0.925,
                0.975
            ]
        },
        # max_grad_norm: float = 0.5,
        # The maximum value for the gradient clipping
        "max_grad_norm": {
            'values': [0.5]
        },

        # learning_rate: Union[float, Schedule] = 3e-4,
        # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
        "learning_rate": {
            'distribution': 'log_uniform',
            'min': math.log(0.00001),
            'max': math.log(0.001)
        },

        # batch_size: Optional[int] = 64,
        # Minibatch size
        "batch_size": {
            'values': [64, 256, 1024]
        },
        # clip_range: Union[float, Schedule] = 0.2,
        # Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
        "clip_range": {
            "distribution": "q_uniform",
            "min": 0.5,
            "max": 0.9,
            "q": 0.125
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
                None,
                0.9,
                0.8
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
            'values': [True, False]
        },
        # n_epochs: int = 10,
        # Number of epoch when optimizing the surrogate loss
        "n_epochs": {
            'values': [
                10,
                50,
                100,
            ]
        },

        # n_steps: int = 2048,
        # The number of steps to run for each environment per update
        # (i.e. batch size is n_steps * n_env where n_env is number of environment
        # copies running in parallel)
        "n_steps": {
            'values': [2048]
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
            'values': [2, 3]
        },
        "net_arch_n_size": {
            "distribution": "q_uniform",
            "min": 20,
            "max": 60,
            "q": 20.0
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
                "Tanh",  # th.nn.Tanh,
                "ReLu",  # th.nn.ReLU
                "Hardtanh",
                "ELU",
                "RRELU"
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
}


def perform_run():
    instance_name = "ft06"
    jsp_instance, jsp_instance_details = env_utils.get_benchmark_instance_and_details(name=instance_name)
    _, n_jobs, n_machines = jsp_instance.shape


    RUN_CONFIG = {
        "total_timesteps": 100_000,
        "n_envs": 8,  # multiprocessing.cpu_count()-1

        "instance_name": instance_name,
        "instance_details": jsp_instance_details,

        "policy_type": MaskableActorCriticPolicy,
        "model_hyper_parameters": {
            "gamma": 0.99,  # discount factor,
            "gae_lambda": 0.95,
            "learning_rate": 3e-4,
            "batch_size": 64,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.0,
            "normalize_advantage": True,
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
            ]
        },

    }

    with wb.init(
            sync_tensorboard=False,
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
            dir=f"{PATHS.WANDB_PATH}/") as run:

        log.info(f"run name: {run.name}, run id: {run.id}")

        sweep_params = wb.config
        log.info(f"hyper params: {pprint.pformat(sweep_params)}")

        # override run config
        model_params = [
            "learning_rate",
            "n_steps",
            "n_epochs",
            "gamma",
            "batch_size",
            "clip_range",
            "clip_range_vf",
            "normalize_advantage",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "target_kl",
            # "tensorboard_log",
            "create_eval_env",
            # "verbose",
            "seed",
            "device"
        ]
        for m_param in model_params:
            RUN_CONFIG["model_hyper_parameters"][m_param] = sweep_params[m_param]

        env_params = [
            "normalize_observation_space",
            "flat_observation_space",
            "perform_left_shift_if_possible",
            "dtype",
            "action_mode",
        ]

        for env_param in env_params:
            RUN_CONFIG["env_kwargs"][env_param] = sweep_params[env_param]


        '''
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        '''
        policy_params = [
            "ortho_init",
            "normalize_images",
        ]
        for p_param in policy_params:
            RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"][p_param] = sweep_params[p_param]

        net_arch = [{
            "pi": [sweep_params["net_arch_n_size"]] * sweep_params["net_arch_n_layers"],
            "vf": [sweep_params["net_arch_n_size"]] * sweep_params["net_arch_n_layers"],
        }]

        raise NotImplementedError("bruh mach implemtierung")
        activation_fn = th.nn.ReLU if sweep_params["activation_fn"] == 'ReLu' else th.nn.Tanh

        RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"]["net_arch"] = net_arch
        RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"]["activation_fn"] = activation_fn
        RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"]["optimizer_kwargs"]["eps"] = sweep_params["optimizer_eps"]

        log.info(f"run config: {pprint.pformat(RUN_CONFIG)}")

        env_kwargs = {
            "jps_instance": jsp_instance,
            "scaling_divisor": jsp_instance_details["lower_bound"],
            **RUN_CONFIG["env_kwargs"]
        }

        log.info(f"setting up vectorised environment")

        def mask_fn(env):
            return env.valid_action_mask()

        venv = make_vec_env_without_monitor(
            env_id=RUN_CONFIG["env_name"],
            env_kwargs=env_kwargs,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn},
            n_envs=RUN_CONFIG["n_envs"]
        )

        venv = CuriosityInfoWrapper(venv=venv)

        venv = VecMonitor(venv=venv)

        log.info(f"setting up mask ppo model")

        model = sb3_contrib.MaskablePPO(
            RUN_CONFIG["policy_type"],
            env=venv,
            verbose=1,
            tensorboard_log=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
            **RUN_CONFIG["model_hyper_parameters"]
        )

        wb_cb = WandbCallback(
            gradient_save_freq=100,
            model_save_path=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
            verbose=1,
        )
        logger_cb = JssLoggerCallback(wandb_ref=wb)

        log.info(f"training the agent")
        model.learn(
            total_timesteps=RUN_CONFIG["total_timesteps"],
            callback=[
                wb_cb,
                logger_cb
            ]
        )

        log.info("evaluating model performance")
        n_eval_episodes = sweep_params["n_eval_episodes"]
        makespans = []

        eval_env_kwargs = env_kwargs.copy()
        eval_env_kwargs["perform_left_shift_if_possible"] = True

        venv = make_vec_env_without_monitor(
            env_id=RUN_CONFIG["env_name"],
            env_kwargs=eval_env_kwargs,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn},
            n_envs=RUN_CONFIG["n_envs"]
        )

        venv = VecMonitor(venv=venv)

        model.set_env(venv)

        for _ in track(range(n_eval_episodes), description="evaluating model performance ..."):
            done = False
            obs = venv.reset()
            while not done:
                masks = np.array([env.action_masks() for env in model.env.envs])
                action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
                obs, rewards, dones, info = venv.step(action)
                done = np.all(dones == True)
                if done:
                    for sub_env_info in info:
                        makespans.append(sub_env_info["makespan"])

        from statistics import mean
        mean_return = mean(makespans)

        log.info(f"mean evaluation makespan: {mean_return:.2f}")
        wb.log({"mean_makespan": mean_return})

        obs = venv.reset()
        venv.close()
        del venv


if __name__ == '__main__':
    sweep_id = wb.sweep(SWEEP_CONFIG, project=PROJECT)
    #sweep_id = "ia0qovaq"
    log.info(f"use this 'sweep_id': '{sweep_id}'")
    wb.agent(
        sweep_id,
        function=perform_run,
        count=1,
        project=PROJECT
    )
