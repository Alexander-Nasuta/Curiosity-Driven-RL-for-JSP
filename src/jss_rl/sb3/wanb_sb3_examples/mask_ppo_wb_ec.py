import gym
import pprint

import numpy as np
import sb3_contrib

import wandb as wb
import torch as th
from rich.progress import track, Progress

import jss_utils.PATHS as PATHS
import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import VecVideoRecorder, VecMonitor
from wandb.integration.sb3 import WandbCallback

from jss_rl.sb3.curiosity.ec_wrapper import EpisodicCuriosityEnvWrapper
from jss_rl.sb3.util.callbacks.episode_end_moving_average_rollout_end_logger_callback import \
    EpisodeEndMovingAverageRolloutEndLoggerCallback
from jss_rl.sb3.util.info_field_moving_avarege_logger_callback import InfoFieldMovingAverageLogger
from jss_rl.sb3.make_vec_env_without_monitor import make_vec_env_without_monitor
from jss_rl.sb3.progress_bar_callback import ProgressBarCallback

from jss_utils.jss_logger import log

PROJECT = "JSP-test"
BENCHMARK_INSTANCE_NAME = "ta01"

wb.tensorboard.patch(root_logdir=str(PATHS.WANDB_PATH))

jsp_instance = parser.get_instance_by_name(BENCHMARK_INSTANCE_NAME)
jsp_instance_details = details.get_jps_instance_details(BENCHMARK_INSTANCE_NAME)

_, n_jobs, n_machines = jsp_instance.shape

gym.envs.register(
    id='GraphJsp-v0',
    entry_point='jss_graph_env.disjunctive_graph_jss_env:DisjunctiveGraphJssEnv',
    kwargs={},
)

config = {
    "total_timesteps": 3_000_000,
    "n_envs": 9,  # multiprocessing.cpu_count()-1

    "instance_name": BENCHMARK_INSTANCE_NAME,
    "instance_details": jsp_instance_details,
    "n_jobs": n_jobs,
    "n_machines": n_machines,

    "policy_type": MaskableActorCriticPolicy,
    "model_hyper_parameters": {
        "gamma": 0.99999,  # discount factor,
        "gae_lambda": 0.95,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "clip_range": 0.541,
        "clip_range_vf": 26,
        "ent_coef": 0.0,
        "normalize_advantage": True,
        # "target_kl": 0.05047, # for early stopping
        "policy_kwargs": {
            "net_arch": [{
                "pi": [128, 128, 128],
                "vf": [128, 128, 128],
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
        "action_mode": 'task',
        "dtype": "float32",
        "verbose": 0,
        "default_visualisations": [
            "gantt_window",
            # "graph_window",  # very expensive
        ]
    },

    "ec_wrapper_kwargs": {

    },

    "EpisodeEndMovingAverageRolloutEndLoggerCallback_kwargs": {
        "fields": [
            "extrinsic_return",
            "bonus_return",
            "total_return",
            "makespan",
            "scaling_divisor",
        ],
        "capacity": 100,
    },
    "InofFieldMovingAvarageLogger_kwargs": {
        "fields": [
            "bonus_reward",
            "extrinsic_reward",
            "ec_loss"
        ],
        "field_capacities": [
            100,
            100,
            100
        ],
    },
    "record_video": False
}

if __name__ == '__main__':
    log.info(f"config: {pprint.pformat(config)}")

    run = wb.init(
        project=PROJECT,
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
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
        "scaling_divisor": jsp_instance_details["lower_bound"],
        **config["env_kwargs"]
    }

    log.info(f"setting up vectorised environment")


    def mask_fn(env):
        return env.valid_action_mask()


    blank_venv = make_vec_env_without_monitor(
        env_id=config["env_name"],
        env_kwargs=env_kwargs,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        n_envs=config["n_envs"]
    )

    venv = EpisodicCuriosityEnvWrapper(
        venv=blank_venv,
        **config["ec_wrapper_kwargs"]
    )

    venv = VecMonitor(
        venv=venv,
        # somehow VecMonitor does not log the info keys properly. Using logger callback instead.
        # info_keywords=("extrinsic_reward","makespan","scaling_divisor",'intrinsic_reward')
    )

    log.info(f"setting up mask ppo model")

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

    episode_logger_cb = EpisodeEndMovingAverageRolloutEndLoggerCallback(
        wandb_ref=wb,
        **config["EpisodeEndMovingAverageRolloutEndLoggerCallback_kwargs"],
    )

    log_field_if_present_logger_cb = InfoFieldMovingAverageLogger(
        wandb_ref=wb,
        **config["InofFieldMovingAvarageLogger_kwargs"],
    )

    log.info(f"training the agent")
    with Progress(transient=True) as progress:
        task1 = progress.add_task("[yellow]training agent", total=config["total_timesteps"])
        pb_cb = ProgressBarCallback(progress=progress, task=task1, total_steps=config["total_timesteps"])
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[wb_cb, episode_logger_cb, log_field_if_present_logger_cb, pb_cb]
        )

    if config["record_video"]:
        # somehow the mask ppo does not work trigger properly. the step appears to count only to the batch size and then
        # start again at step 0
        # therefore here is a workaround
        log.info(f"setting up video recorder")

        _, n_jobs, n_machines = jsp_instance.shape
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
        for _ in track(range(episode_len), description="recording frames ..."):
            masks = np.array([env.action_masks() for env in model.env.envs])
            action, _ = model.predict(observation=obs, deterministic=False, action_masks=masks)
            obs, _, _, infos = venv.step(action)

        for i in range(venv.num_envs):
            wb.log({
                f'video_gantt_df_of_env_{i}': infos[i]["gantt_df"]
            })
        # Save the video
        log.info("saving video...")
        venv.close()

        # video is saved automatically, if monitor_gym=True (see wb.init above)
        # video_file = next(video_folder.glob('*.mp4'))
        # wb_video = wb.Video(data_or_path=str(video_file))
        # wb.log({"video": wb_video})

    # run.finish()
    log.info("done.")
    del venv
