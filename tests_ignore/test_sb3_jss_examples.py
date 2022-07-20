import jss_utils.PATHS as PATHS

from jss_rl.sb3.jss_exsamples.basic_ppo import jss_env_basic_ppo_example
from jss_rl.sb3.jss_exsamples.mask_ppo import jss_env_mask_ppo_example
from jss_rl.sb3.jss_exsamples.mask_ppo_gif import train_and_create_gif
from jss_rl.sb3.jss_exsamples.venv_mask_ppo import venv_basic_ppo_example
from jss_rl.sb3.jss_exsamples.venv_mask_ppo_video import venv_ppo_video_recorder_example


def test_basic_ppo_example(ft06lb):
    jsp, lb = ft06lb
    jss_env_basic_ppo_example(
        jsp_instance=jsp,
        lower_bound=lb,
        total_timesteps=1_000,
        n_eval_episodes=1
    )


def test_mask_ppo_example(ft06lb):
    jsp, lb = ft06lb
    jss_env_mask_ppo_example(
        jsp_instance=jsp,
        lower_bound=lb,
        total_timesteps=1_000
    )


def test_jss_env_gif_of_trained_mask_ppo(ft06lb):
    jsp, lb = ft06lb
    filename = "fancy_jsp"
    train_and_create_gif(
        jsp_instance=jsp,
        lower_bound=lb,
        total_timesteps=1_000,
        filename=filename
    )
    file_path = PATHS.SB3_EXAMPLES_GIF.joinpath(f"{filename}.gif")
    assert file_path.exists()
    assert file_path.is_file()


def test_venv_mask_ppo(ft06env_kwargs):
    venv_basic_ppo_example(
        env_kwargs=ft06env_kwargs,
        total_timesteps=1_000,
        n_envs=4
    )


def test_venv_mask_ppo_video_recorder(ft06env_kwargs):
    venv_ppo_video_recorder_example(
        env_kwargs=ft06env_kwargs,
        total_timesteps=1_000,
        n_envs=2
    )
