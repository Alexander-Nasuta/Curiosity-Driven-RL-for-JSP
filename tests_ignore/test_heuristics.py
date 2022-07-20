from jss_rl.heuristics.analysis import compare_benchmark_results
from jss_utils.jsp_instance_details import download_benchmark_instances_details
from jss_utils.jsp_instance_downloader import download_instances


def test_all_heuristics():
    download_instances(start_id=6, end_id=6)
    download_benchmark_instances_details()
    compare_benchmark_results(name="ft06")