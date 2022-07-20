import pytest

from jss_utils.jsp_env_utils import get_benchmark_instance_and_details
from jss_utils.jsp_instance_details import download_benchmark_instances_details
from jss_utils.jsp_instance_downloader import download_instances
from jss_utils.setup_helper import prepare_for_testing, post_process_testing


@pytest.fixture(autouse=True)
def with_clean_resources_folder():
    # all test with custom instances run on 3x3 instances
    # clear_resources_dir(sure=True)
    prepare_for_testing()

    yield None

    # Code that will run after the tests
    # post_process_testing()


@pytest.fixture(scope="function")
def abz5(with_clean_resources_folder):
    download_instances(start_id=1, end_id=1)
    download_benchmark_instances_details()
    instance, details_dict = get_benchmark_instance_and_details(name="abz5")
    yield instance, details_dict


@pytest.fixture(scope="function")
def ft06(with_clean_resources_folder):
    download_instances(start_id=6, end_id=6)
    download_benchmark_instances_details()
    instance, details_dict = get_benchmark_instance_and_details(name="ft06")
    yield instance, details_dict


@pytest.fixture(scope="function")
def ft06lb(ft06):
    instance, details_dict = ft06
    yield instance, details_dict["lower_bound"]


@pytest.fixture(scope="function")
def ft06env_kwargs(ft06lb):
    instance, lb = ft06lb
    env_kwargs = {
        "jps_instance": instance,
        "scaling_divisor": lb,
        "perform_left_shift_if_possible": True,
        "scale_reward": True,
        "normalize_observation_space": True,
        "flat_observation_space": True,
        "default_visualisations": [
            "gantt_window",
            # "graph_window",  # very expensive
        ],
    }
    yield env_kwargs


@pytest.fixture(scope="session", autouse=True)
def undo_resource_dir_changes():
    yield None
    # Code that will run after all the tests
    # post_process_testing()
