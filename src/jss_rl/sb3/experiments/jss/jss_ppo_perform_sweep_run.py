import pprint
import gym
import sb3_contrib

import wandb as wb
import torch as th
import numpy as np
from rich.progress import track

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import VecMonitor
from wandb.integration.sb3 import WandbCallback

import jss_utils.PATHS as PATHS
import jss_utils.jsp_env_utils as env_utils

from jss_rl.sb3.curiosity_modules.curiosity_info_wrapper import CuriosityInfoWrapper
from jss_rl.sb3.curiosity_modules.ec_wrapper import EpisodicCuriosityModuleWrapper
from jss_rl.sb3.curiosity_modules.icm_wrapper import IntrinsicCuriosityModuleWrapper
from jss_rl.sb3.experiments.jss.jss_dynamic_instance_loader import DynamicCustomInstanceLoaderCallback
from jss_rl.sb3.experiments.jss.jss_logger_cb import JssLoggerCallback
from jss_rl.sb3.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_utils.jsp_env_utils import get_benchmark_instance_and_details

from jss_utils.jss_logger import log

gym.envs.register(
    id='GraphJsp-v0',
    entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
    kwargs={},
)


def perform_jss_run() -> None:
    RUN_CONFIG = {
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

        activation_fn = None
        if sweep_params["activation_fn"] == 'ReLu':
            activation_fn = th.nn.ReLU
        elif sweep_params["activation_fn"] == 'Tanh':
            activation_fn = th.nn.Tanh
        elif sweep_params["activation_fn"] == 'Hardtanh':
            activation_fn = th.nn.Hardtanh
        elif sweep_params["activation_fn"] == 'ELU':
            activation_fn = th.nn.ELU
        elif sweep_params["activation_fn"] == 'RRELU':
            activation_fn = th.nn.ELU
        else:
            raise NotImplementedError(f"activation function '{activation_fn}' is not available/implemented. "
                                      f"You may need to add a case for your activation function")

        RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"]["net_arch"] = net_arch
        RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"]["activation_fn"] = activation_fn
        RUN_CONFIG["model_hyper_parameters"]["policy_kwargs"]["optimizer_kwargs"]["eps"] = sweep_params["optimizer_eps"]

        log.info(f"run config: {pprint.pformat(RUN_CONFIG)}")

        log.info(f"sweep config: {pprint.pformat(sweep_params)}")

        wb_cb = WandbCallback(
            gradient_save_freq=100,
            model_save_path=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
            verbose=1,
        )
        logger_cb = JssLoggerCallback(wandb_ref=wb)

        if not sweep_params["dynamic_instances"]:
            jsp_instance, jsp_instance_details = get_benchmark_instance_and_details(
                name=sweep_params["benchmark_instance"])
            env_kwargs = {
                "jps_instance": jsp_instance,
                "scaling_divisor": jsp_instance_details["lower_bound"],
                **RUN_CONFIG["env_kwargs"]
            }
            callbacks = [wb_cb, logger_cb]
        else:
            jsp_instance, details, _ = env_utils.get_random_custom_instance_and_details_and_name(
                n_jobs=sweep_params["n_jobs"],
                n_machines=sweep_params["n_machines"],
            )
            lb = details["lower_bound"]

            env_kwargs = {
                "jps_instance": jsp_instance,
                "scaling_divisor": lb,
                **RUN_CONFIG["env_kwargs"]
            }

            dil_cb = DynamicCustomInstanceLoaderCallback(
                load_instance_every_n_rollouts=sweep_params["load_instance_every_n_rollouts"],
                verbose=1,
            )

            callbacks = [wb_cb, dil_cb, logger_cb]

        log.info(f"setting up vectorised environment")

        def mask_fn(env):
            return env.valid_action_mask()

        venv = make_vec_env_without_monitor(
            env_id=RUN_CONFIG["env_name"],
            env_kwargs=env_kwargs,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn},
            n_envs=sweep_params["n_envs"]
        )

        if sweep_params["curiosity_module"] is None:
            venv = CuriosityInfoWrapper(venv=venv)
        elif sweep_params["curiosity_module"] == "icm":

            feature_nets_arch = [sweep_params["feature_nets_arch_n_size"]] * sweep_params["feature_nets_n_layers"]

            forward_net_arch = [sweep_params["forward_net_arch_n_size"]] * sweep_params["forward_net_n_layers"]

            icm_defaults = {
                "beta": 0.2,
                "eta": 1.0,
                "lr": 1e-3,
                "feature_dim": 288,
                "feature_net_activation": "relu",
                "inverse_feature_net_activation": "relu",
                "forward_fcnet_net_activation": "relu",

                "memory_capacity": 100,
                "clear_memory_on_end_of_episode": False,
                "clear_memory_every_n_steps": None,
                "shuffle_samples": True,
                "maximum_sample_size": None,
                "postprocess_on_end_of_episode": True,
                "postprocess_every_n_steps": None,
                "exploration_steps": None,
            }

            icm_params = {key: sweep_params.get(key, val) for key, val in icm_defaults.items()}

            # set archs
            icm_params["feature_net_hiddens"] = feature_nets_arch
            icm_params["inverse_feature_net_hiddens"] = feature_nets_arch
            icm_params["forward_fcnet_net_hiddens"] = forward_net_arch

            if sweep_params["maximum_sample_size_pct"]:
                icm_params["maximum_sample_size"] = int(
                    icm_params["memory_capacity"] * sweep_params["maximum_sample_size_pct"]
                )

            # icm_memory_clearing
            # icm_memory_clearing
            # postprocess_on_end_of_episode
            # feature_nets_n_layers
            # forward_net_arch_n_size
            # forward_net_n_layers
            if sweep_params["icm_postprocess_trigger"] == "step":
                # disable 'episode based' params
                icm_params["postprocess_on_end_of_episode"] = False
                icm_params["clear_memory_on_end_of_episode"] = False

                # set 'step based' params
                icm_params["postprocess_every_n_steps"] = sweep_params["postprocess_every_n_steps"]
                if sweep_params["icm_memory_clearing"]:
                    icm_params["clear_memory_every_n_steps"] = sweep_params["clear_memory_every_n_steps"]
            elif sweep_params["icm_postprocess_trigger"] == "episode":
                # disable 'step based' params
                icm_params["postprocess_every_n_steps"] = None
                icm_params["clear_memory_every_n_steps"] = None
                # set 'episode based' params
                icm_params["postprocess_on_end_of_episode"] = True
                icm_params["clear_memory_on_end_of_episode"] = sweep_params["icm_memory_clearing"]
            else:
                raise ValueError(f"'icm_postprocess_trigger' must be 'step' or 'episode' "
                                 f"not '{sweep_params['icm_postprocess_trigger']}'.")

            log.info(f"icm params: {pprint.pformat(icm_params)}")

            venv = IntrinsicCuriosityModuleWrapper(
                venv=venv,
                **icm_params,
            )
        elif sweep_params["curiosity_module"] == "ec":

            ec_defaults = {
                "embedding_net_hiddens": None,
                "embedding_net_activation": "relu",
                "comparator_net_hiddens": None,
                "comparator_net_activation": "relu",
                "alpha": 1.0,
                "beta": 1.0,
                "lr": 1e-3,
                "k": 2,
                "gamma": 3,
                "b_novelty": 0.0,
                "episodic_memory_capacity": 100,
                "clear_memory_every_episode": True,
                "exploration_steps": None,
            }

            ec_params = {key: sweep_params.get(key, val) for key, val in ec_defaults.items()}
            # gamma is also a param of mask ppo. Therefore 'ec_gamma' is used in the sweep config
            ec_params["gamma"] = sweep_params["ec_gamma"]

            comparator_net_arch = [sweep_params["comparator_net_arch_n_size"]] * sweep_params["comparator_net_n_layers"]
            embedding_net_arch = [sweep_params["embedding_net_arch_n_size"]] * sweep_params["embedding_net_n_layers"]

            ec_params["comparator_net_hiddens"] =comparator_net_arch
            ec_params["embedding_net_hiddens"] = embedding_net_arch

            log.info(f"ec params: {pprint.pformat(ec_params)}")

            venv = EpisodicCuriosityModuleWrapper(
                venv=venv,
                **ec_params
            )
        else:
            raise ValueError(f" '{sweep_params['curiosity_module']}' is not a valid argument for 'curiosity_module'. "
                             f" Valid 'curiosity_module'-params are : [None, 'icm', 'ec']")

        venv = VecMonitor(venv=venv)

        log.info(f"setting up mask ppo model")

        model = sb3_contrib.MaskablePPO(
            RUN_CONFIG["policy_type"],
            env=venv,
            verbose=1,
            tensorboard_log=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
            **RUN_CONFIG["model_hyper_parameters"]
        )

        log.info(f"training the agent")
        model.learn(total_timesteps=sweep_params["total_timesteps"], callback=callbacks)

        log.info("evaluating model performance")
        n_eval_episodes = sweep_params["n_eval_episodes"]
        makespans = []

        jsp_instance, jsp_instance_details = get_benchmark_instance_and_details(name=sweep_params["benchmark_instance"])

        eval_env_kwargs = env_kwargs.copy()
        eval_env_kwargs["perform_left_shift_if_possible"] = True
        eval_env_kwargs["jps_instance"] = jsp_instance
        eval_env_kwargs["scaling_divisor"] = jsp_instance_details["lower_bound"]

        venv = make_vec_env_without_monitor(
            env_id=RUN_CONFIG["env_name"],
            env_kwargs=eval_env_kwargs,
            wrapper_class=ActionMasker,
            wrapper_kwargs={"action_mask_fn": mask_fn},
            n_envs=sweep_params["n_envs"]
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
    # sweep_id = "9fknmeo9" # random sweep
    sweep_id = "3n6q34m3"  # grid sweep
    wb.agent(
        sweep_id,
        function=perform_jss_run,
        count=1,
        project="test"
    )
