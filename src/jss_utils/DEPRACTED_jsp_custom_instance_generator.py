"""
the code in this file based on a script witten by Vladimir Samsonov
"""

import collections
import json

import pandas as pd
import numpy as np
import pathlib as pl

import jss_utils.PATHS as PATHS

from ortools.sat.python import cp_model
from typing import List, Union

from jss_utils.jss_logger import log


def generate_jsp(n_jobs: int = 2, n_resources: int = 2, n_ops_per_job: int = 2, max_ops_time: int = 11) \
        -> (pd.DataFrame, List):
    log.info(f"generating a jsp of size ({n_jobs},{n_resources}). "
             f"(ops_per job: {n_ops_per_job}, max_ops_time: {max_ops_time})")

    machine_strings = [str(m) for m in range(n_resources)]
    df = pd.DataFrame(columns=["job", "order"])
    df["job"] = np.repeat(np.arange(0, n_jobs), n_ops_per_job)
    df["order"] = np.tile(np.arange(0, n_ops_per_job), n_jobs)
    or_solver = []

    for m in range(n_resources):
        df[machine_strings[m]] = 0

    for j in range(n_jobs):
        job_data_or_solver = []
        mach_nums = np.arange(0, n_resources)

        for o in range(n_ops_per_job):
            index = np.random.randint(0, mach_nums.size)
            mach_num = mach_nums[index]
            mach_nums = np.delete(mach_nums, index)
            time = np.random.randint(1, max_ops_time + 1)
            df.loc[(df.job == j) & (df.order == o), [str(mach_num)]] = time
            job_data_or_solver.append([mach_num, time])
        or_solver.append(job_data_or_solver)
    # print(df)
    return df, or_solver


def solve_jsp_or_tools(jobs_data: List) -> (float, str, pd.DataFrame, str):
    log.info("solving jsp with google or tools...")
    # Create the model.
    model = cp_model.CpModel()

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = f'_{job_id}_{task_id}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.Value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1]))

        # Create per machine output lines.
        output = ''
        gantt_data = []

        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-10s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-10s' % sol_tmp

                # add Job_num,Mach_num,Job_time,Start_time
                gantt_data.append([assigned_task.job, machine, duration, start])

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        gantt_data = pd.DataFrame(data=gantt_data, columns=["Job_num", "Mach_num", "Job_time", "Start_time"])

        # Finally, print the solution found.
        optimal_time = solver.ObjectiveValue()

        status = 'OPTIMAL' if cp_model.OPTIMAL else 'FEASIBLE'
        log.info(f"or tools: {status} Schedule Length: {optimal_time}")

        return optimal_time, output, gantt_data, status
    else:
        log.error("could not find a feasible schedule.")
        raise RuntimeError("could not find a feasible schedule.")


def generate_jsp_instances(n_jsp_instances: int = 1, generate_job_kwargs=None) -> None:
    if generate_job_kwargs is None:
        generate_job_kwargs = {
            "max_ops_time": 14,
            "n_jobs": 3,
            "n_resources": 3,
            "n_ops_per_job": 3
        }

    save_dir = PATHS.JSP_INSTANCES_CUSTOM_PATH.joinpath(
        f"{generate_job_kwargs['n_jobs']}x{generate_job_kwargs['n_resources']}"
    )
    save_dir.mkdir(exist_ok=True, parents=True)

    n_instances_in_dir = sum([1 for _ in save_dir.glob('*.json')])

    for i in range(n_instances_in_dir, n_instances_in_dir + n_jsp_instances):
        df, or_solver = generate_jsp(**generate_job_kwargs)
        or_tools_time, output, gantt_data, or_status = solve_jsp_or_tools(jobs_data=or_solver)

        # save to json
        machine_list = [col for col in df.columns if col.isdigit()]
        n_jobs = df.job.nunique()  # number of jobs to be done
        n_resources = len(machine_list)  # number of machines
        n_ops_per_job = df.groupby(
            'job').order.nunique().max()  # number of operations per job (referred as orders here...)
        max_op_time = df.groupby('job').max().loc[:, machine_list].max().max()

        jssp_identification = f"{generate_job_kwargs['n_jobs']}x{generate_job_kwargs['n_resources']}_{i}_inst.json"

        jps_inst_file = save_dir.joinpath(jssp_identification)

        print(df)

        jssp_dict = {
            'or_tools': or_tools_time,
            'or_status': or_status,
            'n_jobs': n_jobs,
            'n_resources': n_resources,
            'n_ops_per_job': int(n_ops_per_job),
            'max_op_time': int(max_op_time),
            'jssp_identification': jssp_identification,
            "jssp_instance": {
                "durations": [],
                "machines": [],
            }
        }

        log.info(f"saving generated jsp to '{jps_inst_file}'")
        with open(jps_inst_file, 'w') as f:
            json.dump(jssp_dict, f, indent=4)


if __name__ == '__main__':
    # df, or_solver = generate_jsp()
    # solve_jsp_or_tools(or_solver)
    generate_jsp_instances()
