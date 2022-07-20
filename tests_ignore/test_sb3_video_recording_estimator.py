from jss_rl.sb3.video_cost_estimator import estimate_video_recording_costs


def test_video_cost_estimator(ft06env_kwargs):
    estimate_video_recording_costs(
        env_kwargs=ft06env_kwargs,
        n_recorder_envs=1,
        video_len=3,
    )