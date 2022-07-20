import pprint
import gym
import sb3_contrib

import wandb as wb
import numpy as np

import jss_utils.PATHS as PATHS
import jss_utils.jsp_env_utils as env_utils

from typing import Dict

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from jss_rl.sb3.curiosity_modules.curiosity_info_wrapper import CuriosityInfoWrapper
from jss_rl.sb3.experiments.jss.jss_example_default_config import jss_default_config
from jss_rl.sb3.experiments.jss.jss_logger_cb import JssLoggerCallback
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_utils.jss_logger import log


gym.envs.register(
    id='GraphJsp-v0',
    entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
    kwargs={},
)


def run_jss_masked_ppo_dynamic_instance_example(total_timesteps: int = 100_000, *,
                               jsp_instance: np.ndarray,
                               jsp_instance_details: Dict,
                               is_benchmark_instance: bool,
                               project: str,
                               instance_name: str = None,
                               scaling_divisor: float = 1.0,
                               group="PPO",
                               additional_config: Dict = None) -> None:
    if additional_config is None:
        additional_config = {}

    _, n_jobs, n_machines = jsp_instance.shape

    config = jss_default_config.copy()
    config = {
        **config,
        **additional_config,

        "jsp_instance": jsp_instance,
        "jsp_instance_details": jsp_instance_details,
        "instance_name": instance_name,
        "is_benchmark_instance": is_benchmark_instance,
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "scaling_divisor": scaling_divisor,

        "total_timesteps": total_timesteps,
        "curiosity_module": "None",
        "agent_algorithm": "masked PPO",
        "group_arg": group,
    }

    log.info(f"['Mask PPO'] config: {pprint.pformat(config)}")

    run = wb.init(
        project=project,
        group="Mask PPO",
        config=config,
        sync_tensorboard=config["sync_tensorboard"],
        monitor_gym=config["monitor_gym"],  # auto-upload the videos of agents playing the game
        save_code=config["save_code"],  # optional
        dir=f"{PATHS.WANDB_PATH}/",
    )

    log.info(f"run name: {run.name}, run id: {run.id}")

    wb.log({
        "tasks_to_machines_mapping": wb.Table(
            data=jsp_instance[0],
            columns=[f"task #{i}" for i in range(n_machines)]
        )
    })

    wb.log({
        "tasks_to_duration_mapping": wb.Table(
            data=jsp_instance[1],
            columns=[f"task #{i}" for i in range(n_machines)]
        )
    })

    env_kwargs = {
        "jps_instance": jsp_instance,
        "scaling_divisor": config["scaling_divisor"],
        **config["env_kwargs"]
    }

    def mask_fn(env):
        return env.valid_action_mask()

    venv = make_vec_env_without_monitor(
        env_id=config["env_name"],
        env_kwargs=env_kwargs,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        n_envs=config["n_envs"]
    )

    venv = CuriosityInfoWrapper(venv=venv)

    venv = VecMonitor(venv=venv)

    model = sb3_contrib.MaskablePPO(
        config["policy_type"],
        env=venv,
        verbose=1,
        tensorboard_log=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
        **config["model_hyper_parameters"]
    )

    wb_cb = WandbCallback(
        gradient_save_freq=100,
        model_save_path=PATHS.WANDB_PATH.joinpath(f"{run.name}_{run.id}"),
        verbose=1,
    )

    logger_cb = JssLoggerCallback(
        wandb_ref=wb
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wb_cb, logger_cb]
    )

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
    log.info("logging video to wandb video...")
    venv.close()

    # video is saved automatically, if monitor_gym=True (see wb.init above)
    # video_file = next(video_folder.glob('*.mp4'))
    # wb_video = wb.Video(data_or_path=str(video_file))
    # wb.log({"video": wb_video})

    run.finish()
    del venv


if __name__ == '__main__':
    wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))

    jsp_instance, jsp_instance_details = env_utils.get_benchmark_instance_and_details(name="ft06")

    run_jss_masked_ppo_dynamic_instance_example(
        project="test",
        jsp_instance=jsp_instance,
        jsp_instance_details=jsp_instance_details,
        is_benchmark_instance=True,
        scaling_divisor=jsp_instance_details["lower_bound"],
        instance_name="ft06",
        total_timesteps=50_000,
    )