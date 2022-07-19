import jss_utils.jsp_env_utils as env_utils
import jss_utils.jsp_or_tools_solver as or_tools_solver

import jss_rl.heuristics.time_domain.STT_shortest_task_time as STT
import jss_rl.heuristics.time_domain.SRPT_shortest_remaining_processing_time as SRPT
import jss_rl.heuristics.time_domain.SPT_shortest_processing_time as SPT
import jss_rl.heuristics.time_domain.SNRT_smallest_number_of_remaining_tasks as SNRT
import jss_rl.heuristics.time_domain.random as random_heuristic
import jss_rl.heuristics.time_domain.LTT_longest_task_time as LTT
import jss_rl.heuristics.time_domain.LTS_longest_task_successor as LTS
import jss_rl.heuristics.time_domain.LRPT_longest_remaining_processing_time as LRPT
import jss_rl.heuristics.time_domain.LPT_longest_processing_time as LPT
import jss_rl.heuristics.time_domain.LNRT_largest_number_of_remaining_tasks as LNRT
import jss_rl.heuristics.time_domain.FCFS_first_come_first_serve as FCFS

import jss_rl.heuristics.graph_domain.G_random as g_random_heuristic
import jss_rl.heuristics.graph_domain.GSTT_shortest_task_time_first as GSTT
import jss_rl.heuristics.graph_domain.GSRPT_shortest_remaining_processing_time as GSRPT
import jss_rl.heuristics.graph_domain.GLTT_longest_task_time_first as GLTT
import jss_rl.heuristics.graph_domain.GLRPT_longest_remaining_processing_time as GLRPT
import jss_rl.heuristics.graph_domain.GLNRT_largest_number_of_remaining_tasks as GLNRT

from tabulate import tabulate

from jss_utils.jss_logger import log


def compare_benchmark_results(name: str) -> None:
    jsp, details = env_utils.get_benchmark_instance_and_details(name=name)

    log.info(f"comparing different methods on instance '{name}'")

    _, n_jobs, n_machines = jsp.shape

    # safety check
    assert n_jobs == details["jobs"]
    assert n_machines == details["machines"]

    lower_bound = details["lower_bound"]
    upper_bound = details["upper_bound"]
    lb_optimal = details["lb_optimal"]

    headers = ['', 'makespan', 'scaled', 'solving duration', 'comment']

    table_data = []

    lit_makespan = lower_bound if lower_bound == upper_bound else None
    optimal_makespan = lit_makespan if lb_optimal else None

    # literature data
    table_data.append(
        [
            'literature',
            lit_makespan if lit_makespan else '-',
            lit_makespan / optimal_makespan if optimal_makespan else '-',
            '-',
            'optimal solution' if lb_optimal else ''
        ]
    )

    # or tools
    makespan, status, _, info = or_tools_solver.solve_jsp(jsp_instance=jsp, plot_results=False)

    table_data.append(
        [
            'OR tools',
            lit_makespan if lit_makespan else '-',
            f'{lit_makespan / optimal_makespan:.2f}' if optimal_makespan else '-',
            f"{info['solving_duration']:.2f}",
            'optimal solution' if status == "OPTIMAL" else ''
        ]
    )

    # heuristics
    heuristics = [
        ("STT", STT.solve_jsp),
        ("SRPT", SRPT.solve_jsp),
        ("SPT", SPT.solve_jsp),
        ("SNRT", SNRT.solve_jsp),
        ("Random", random_heuristic.solve_jsp),
        ("LTT", LTT.solve_jsp),
        ("LTS", LTS.solve_jsp),
        ("LRPT", LRPT.solve_jsp),
        ("LPT", LPT.solve_jsp),
        ("LNRT", LNRT.solve_jsp),
        ("FCFS", FCFS.solve_jsp),

        ("G_Random", g_random_heuristic.solve_jsp),
        ("GSTT", GSTT.solve_jsp),
        ("GSRPT", GSRPT.solve_jsp),
        ("GLTT", GLTT.solve_jsp),
        ("GLRPT", GLRPT.solve_jsp),
        ("GLNRT", GLNRT.solve_jsp),
    ]

    for h_name, solve_fct in heuristics:
        log.info(f"solving jsp with '{h_name}' heuristic")
        makespan, info = solve_fct(jsp_instance=jsp, plot_results=False)
        table_data.append(
            [
                h_name,
                makespan,
                f'{makespan / optimal_makespan:.2f}' if optimal_makespan else '-',
                f"{info['solving_duration']:.2f}",
                ''
            ]
        )

    log.info("rendering table...")
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


if __name__ == '__main__':
    compare_benchmark_results("ft06")
