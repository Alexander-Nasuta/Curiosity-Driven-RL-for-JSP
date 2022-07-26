from jss_rl.sb3.curiosity_modules.curiosity_info_wrapper_cartpole_example import curiosity_info_wrapper_example


def test_curiosity_infor_wrapper_on_cartpole():
    curiosity_info_wrapper_example(total_timesteps=1_000, n_envs=2)