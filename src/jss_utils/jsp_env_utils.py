import numpy as np

import jss_utils.jsp_instance_parser as parser
import jss_utils.jsp_instance_details as details

from typing import Union

from stable_baselines3.common.vec_env import VecEnv

from jss_graph_env.disjunctive_graph_jss_env import DisjunctiveGraphJssEnv
from jss_utils.jss_logger import log


def get_benchmark_instance_and_lower_bound(name: str) -> (np.array, float):
    jsp_instance_details = details.get_jps_instance_details(name)
    jps_instance = parser.get_instance_by_name(name)
    return jps_instance, jsp_instance_details["lower_bound"]


def get_pre_configured_example_env(name="ft6", **kwargs) -> DisjunctiveGraphJssEnv:
    env = DisjunctiveGraphJssEnv(**kwargs)
    return load_benchmark_instance_to_environment(env=env, name=name)


def load_benchmark_instance_to_environment(env: Union[DisjunctiveGraphJssEnv, VecEnv, None] = None, name: str = None) \
        -> Union[DisjunctiveGraphJssEnv, VecEnv]:
    if name is None:
        name = "ft06"
        log.info(f"no benchmark is specified. Usinig '{name}' as a fallback.")

    all_instance_details_dict = details.parse_instance_details()

    if name not in all_instance_details_dict.keys():
        error_msg = f"the instance {name} is not present in the details dict. " \
                    f"you might need to download the all benchmark instance and download benchmark details first. " \
                    f"try to run the 'jss_utils.jsp_instance_downloader' script. " \
                    f"And then the 'jss_utils.jsp_instance_details' script. "
        log.error(error_msg)
        raise RuntimeError(error_msg)

    jsp_instance_details = all_instance_details_dict[name]
    jps_instance = parser.get_instance_by_name(name)

    if env is None:
        log.info("no environment is specified. Creating a blank environment with default parameters")
        env = DisjunctiveGraphJssEnv()

    if isinstance(env, DisjunctiveGraphJssEnv):
        log.info(f"loading instance '{name}' into the environment.")
        env.load_instance(
            jsp_instance=jps_instance,
            scaling_divisor=jsp_instance_details["lower_bound"]
        )
        return env
    elif isinstance(env, VecEnv):
        raise NotImplementedError()
    else:
        error_msg = f"the specified environment type ({type(env)}) is not supported."
        log.error(error_msg)
        raise ValueError(error_msg)


if __name__ == '__main__':
    load_benchmark_instance_to_environment()
