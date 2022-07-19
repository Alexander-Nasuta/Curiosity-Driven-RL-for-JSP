from jss_utils.jsp_instance_details import download_benchmark_instances_details, \
    get_jps_instance_details
from jss_utils.jsp_instance_downloader import download_instances


def test_benchmark_instance_downloader_single():
    # abz5
    download_instances(start_id=1, end_id=1)
    download_instances(start_id=1, end_id=1)


def test_download_details():
    download_benchmark_instances_details()
    details_dict = get_jps_instance_details("abz6")

    # "abz6": {
    #         "lower_bound": 943,
    #         "upper_bound": 943,
    #         "jobs": 10,
    #         "machines": 10,
    #         "n_solutions": 2159,
    #         "lb_optimal": true
    #     },
    assert details_dict["lower_bound"] == 943
    assert details_dict["upper_bound"] == 943
    assert details_dict["jobs"] == 10
    assert details_dict["machines"] == 10
    assert details_dict["lb_optimal"]
