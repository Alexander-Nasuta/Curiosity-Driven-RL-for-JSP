import numpy as np

from jss_utils import PATHS
from jss_utils.jsp_instance_downloader import download_instances
from jss_utils.jsp_instance_parser import parse_jps_standard_specification, parse_jps_taillard_specification, \
    get_instance_by_name


def test_parser():
    # abz5
    download_instances(start_id=1, end_id=1)

    abz5_std_path = PATHS.JSP_INSTANCES_STANDARD_PATH.joinpath("abz5.txt")
    abz5_ta_path = PATHS.JSP_INSTANCES_TAILLARD_PATH.joinpath("abz5.txt")

    jsp_instance_from_std, std_matrix = parse_jps_standard_specification(abz5_std_path)
    jsp_instance_from_ta, taillard_matrix = parse_jps_taillard_specification(abz5_ta_path)

    assert np.array_equal(jsp_instance_from_std, jsp_instance_from_ta)

    jsp_by_name = get_instance_by_name("abz5")

    assert np.array_equal(jsp_instance_from_std, jsp_by_name)