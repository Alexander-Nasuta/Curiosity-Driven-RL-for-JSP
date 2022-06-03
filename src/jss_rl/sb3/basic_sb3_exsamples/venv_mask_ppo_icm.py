import sb3_contrib
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.vec_env import VecMonitor

import jss_utils.jsp_env_utils as env_utils
from jss_rl.sb3.curiosity.icm_wrapper import IntrinsicCuriosityModuleWrapper
from jss_rl.sb3.util.make_vec_env_without_monitor import make_vec_env_without_monitor

from jss_utils.jss_logger import log
from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv

from sb3_contrib.common.wrappers import ActionMasker


if __name__ == '__main__':

    jsp, lb = env_utils.get_benchmark_instance_and_lower_bound("ft06")

    env_kwargs = {
        # placeholder for action and observation space shape
        "jps_instance": jsp,
        "scaling_divisor": lb,
        "perform_left_shift_if_possible": True,
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True
    }

    log.info("setting up vectorised environment")


    def mask_fn(env):
        return env.valid_action_mask()

    venv = make_vec_env_without_monitor(
        env_id=DisjunctiveGraphJssEnv,
        env_kwargs=env_kwargs,

        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},

        n_envs=4)

    icm_env = IntrinsicCuriosityModuleWrapper(venv=venv)

    icm_env = VecMonitor(venv=icm_env)

    model = sb3_contrib.MaskablePPO(
        MaskableActorCriticPolicy,
        env=icm_env,
        verbose=1,
    )

    model.learn(total_timesteps=25_000)


