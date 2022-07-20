from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs

class DistanceWrapper(VecEnvWrapper):

    def __init__(self, venv):
        VecEnvWrapper.__init__(self, venv=venv)

    def step_wait(self) -> VecEnvStepReturn:
        """Overrides VecEnvWrapper.step_wait."""
        observations, rewards, dones, infos = self.venv.step_wait()

        for i, o in enumerate(observations):
            x, y = o % 8, o // 8  # frozen lake with 8x8 size
            infos[i]["distance_from_origin"] = (x ** 2 + y ** 2) ** 0.5

        return observations, rewards, dones, infos

    def reset(self) -> VecEnvObs:
        """Overrides VecEnvWrapper.reset."""
        observations = self.venv.reset()
        return observations
