import pprint

import numpy as np
import sb3_contrib
import gym
import wandb as wb
import torch as th
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

import jss_utils.PATHS as PATHS
from jss_rl.sb3.curiosity_modules.ec_wrapper import EpisodicCuriosityModuleWrapper

from jss_rl.sb3.experiments.jss.jss_logger_cb import JssLoggerCallback
from jss_rl.sb3.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_utils.jsp_env_utils import get_benchmark_instance_and_details

from jss_utils.jss_logger import log


def run_ppo_ec_jss_experiment_6x6(total_timesteps: int, *, instance_name: str,
                           project: str, group="PPO + EC", additional_config=None):
    if additional_config is None:
        additional_config = {}

    jsp_instance, jsp_instance_details = get_benchmark_instance_and_details(name=instance_name)
    _, n_jobs, n_machines = jsp_instance.shape
    env_config = {
        "env_name": "GraphJsp-v0",

        "n_envs": 8,
        "instance_name": instance_name,
        "is_benchmark_instance": True,
        "n_jobs": n_jobs,
        "n_machines": n_machines,

        "env_kwargs": {
            "scale_reward": True,
            "normalize_observation_space": True,
            "flat_observation_space": True,
            "perform_left_shift_if_possible": True,
            "dtype": "float32",
            "action_mode": "task",
            "jps_instance": jsp_instance,
            "scaling_divisor": jsp_instance_details["lower_bound"],
            "default_visualisations": [
                "gantt_window",
                # "graph_window",  # very expensive
            ]
        }
    }
    model_config = {
        "policy_type": MaskableActorCriticPolicy,

        "model_hyper_parameters": {
            "gamma": 0.99013,  # discount factor,
            "gae_lambda": 0.9,
            "batch_size": 64,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "vf_coef": 0.5,
            "ent_coef": 0.0,
            "normalize_advantage": True,
            "n_epochs": 28,
            "n_steps": 432,
            "device": "auto",
            "max_grad_norm": 0.5,
            "learning_rate": 6e-4,
            "seed": None,
            "create_eval_env": False,
            "target_kl": None,
            "policy_kwargs": {
                "net_arch": [{
                    "pi": [90, 90],
                    "vf": [90, 90],
                }],
                "ortho_init": True,
                "activation_fn": th.nn.ELU,
                "normalize_images": True,
                "optimizer_kwargs": {  # for th.optim.Adam
                    "eps": 1e-7
                }
            }
        }
    }
    ec_config = {
        "EpisodicCuriosityModuleWrapper": {
            "alpha": 0.0025,
            "beta": 0.5,
            "gamma": 2,
            "embedding_dim": 288,
            "episodic_memory_capacity": 500,
            "clear_memory_every_episode": False,
            "exploration_steps": total_timesteps * 30_000 / 40_000,
        }
    }
    config = {
        **model_config,
        **env_config,
        **ec_config,
        **additional_config,
        "total_timesteps": total_timesteps,
        "curiosity_module": "ec",
        "agent_algorithm": "PPO + EC",
        "group_arg": group,
    }

    log.info(f"run config: {pprint.pformat(config)}")

    run = wb.init(
        project=project,
        group=group,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        dir=f"{PATHS.WANDB_PATH}/",
    )

    wb_cb = WandbCallback(
        gradient_save_freq=100,
        model_save_path=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
        verbose=1,
    )
    logger_cb = JssLoggerCallback(wandb_ref=wb)

    callbacks = [wb_cb, logger_cb]

    log.info(f"setting up vectorised environment")

    def mask_fn(env):
        return env.valid_action_mask()

    venv = make_vec_env_without_monitor(
        env_id=config["env_name"],
        env_kwargs=config["env_kwargs"],
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        n_envs=config["n_envs"]
    )

    venv = EpisodicCuriosityModuleWrapper(
        venv=venv,
        **config["EpisodicCuriosityModuleWrapper"]
    )
    venv = VecMonitor(venv=venv)

    log.info(f"setting up mask ppo model")

    model = sb3_contrib.MaskablePPO(
        config["policy_type"],
        env=venv,
        verbose=1,
        tensorboard_log=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
        **config["model_hyper_parameters"]
    )

    log.info(f"training the agent")
    model.learn(total_timesteps=config["total_timesteps"], callback=callbacks)

    log.info(f"setting up video recorder")

    episode_len = n_jobs * n_machines

    video_folder = PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}")

    venv = VecVideoRecorder(
        venv=venv,
        video_folder=video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=episode_len,
        name_prefix=f"{run.name}_{run.id}")

    obs = venv.reset()
    infos = None

    log.info("recording frames ...")
    for _ in range(episode_len):
        masks = np.array([env.action_masks() for env in model.env.envs])
        action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
        obs, _, _, infos = venv.step(action)

    for i in range(venv.num_envs):
        wb.log({
            f'gantt_df_of_env_{i}': infos[i]["gantt_df"]
        })

    # Save the video
    log.info("logging video to wandb ...")
    venv.close()

    # video is saved automatically, if monitor_gym=True (see wb.init above)
    # video_file = next(video_folder.glob('*.mp4'))
    # wb_video = wb.Video(data_or_path=str(video_file))
    # wb.log({"video": wb_video})

    run.finish()
    del venv


if __name__ == '__main__':
    gym.envs.register(
        id='GraphJsp-v0',
        entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
        kwargs={},
    )
    wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))
    run_ppo_ec_jss_experiment_6x6(
        total_timesteps=60_000,
        instance_name="ft06",
        project="test"
    )
