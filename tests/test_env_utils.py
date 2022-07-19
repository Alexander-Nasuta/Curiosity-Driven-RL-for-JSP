import numpy as np

import jss_utils.PATHS as PATHS
from jss_utils.jsp_custom_instance_generator import generate_jsp_instances

from jss_utils.jsp_env_utils import get_benchmark_instance_and_lower_bound, get_benchmark_instance_and_details, \
    get_pre_configured_example_env, load_benchmark_instance_to_environment, \
    get_random_custom_instance_and_details_and_name
from jss_utils.jsp_instance_details import download_benchmark_instances_details
from jss_utils.jsp_instance_downloader import download_instances
from jss_utils.jsp_instance_parser import parse_jps_taillard_specification


def test_instance_lb_loader():
    download_instances(start_id=1, end_id=1)
    download_benchmark_instances_details()

    abz5_ta_path = PATHS.JSP_INSTANCES_TAILLARD_PATH.joinpath("abz5.txt")
    jsp_instance_from_ta, _ = parse_jps_taillard_specification(abz5_ta_path)
    instance, lb = get_benchmark_instance_and_lower_bound(name="abz5")

    assert np.array_equal(instance, jsp_instance_from_ta)
    assert lb == 1234


def test_instance_details_loader():
    download_instances(start_id=2, end_id=2)
    download_benchmark_instances_details()

    abz6_ta_path = PATHS.JSP_INSTANCES_TAILLARD_PATH.joinpath("abz6.txt")
    jsp_instance_from_ta, _ = parse_jps_taillard_specification(abz6_ta_path)
    instance, details_dict = get_benchmark_instance_and_details(name="abz6")

    assert np.array_equal(instance, jsp_instance_from_ta)
    assert details_dict["lower_bound"] == 943
    assert details_dict["upper_bound"] == 943
    assert details_dict["jobs"] == 10
    assert details_dict["machines"] == 10
    assert details_dict["lb_optimal"] == True


def test_default_env():
    download_instances(start_id=3, end_id=3)
    download_instances(start_id=6, end_id=6)
    download_benchmark_instances_details()

    env = get_pre_configured_example_env(name="abz7")
    env.step(0)
    env.step(0)

    env = get_pre_configured_example_env(name="ft06")
    env.step(0)
    env.step(0)


def test_load_benchmark_instance_to_environment():
    download_instances(start_id=1, end_id=2)
    download_benchmark_instances_details()
    env = get_pre_configured_example_env(name="abz5")
    load_benchmark_instance_to_environment(env=env, name="abz6")


def test_get_random_custom_instance_and_details_and_name():
    generate_jsp_instances(
        n_instances=2,
        n_jobs=3,
        n_machines=3
    )
    get_random_custom_instance_and_details_and_name(n_jobs=3, n_machines=3)
