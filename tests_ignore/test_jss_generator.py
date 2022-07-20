from jss_utils.jsp_custom_instance_generator import generate_jsp_instances
from jss_utils.jsp_env_utils import get_random_custom_instance_and_details_and_name


def test_jsp_generator():
    generate_jsp_instances(
        n_instances=2,
        n_jobs=3,
        n_machines=3
    )
    get_random_custom_instance_and_details_and_name(n_jobs=3, n_machines=3)
