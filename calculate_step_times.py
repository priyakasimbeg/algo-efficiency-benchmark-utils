import log_utils
import os
import pandas as pd

LOG_DIR = 'step_time_logs'

def get_workload_name_from_logfilename(logfile):
    filename = os.path.basename(logfile)
    if 'jax' in filename:
        workload = (filename.split("_jax")[0]).split("_", 1)[1]
    elif 'pytorch' in filename:
        workload = (filename.split("_pytorch")[0]).split("_", 1)[1]
    return workload

def get_algo_name_from_logfilename(logfile):
    filename = os.path.basename(logfile)
    if 'jax' in filename:
        algo = (filename.split("_jax")[0]).split("_", 1)[0]
    elif 'pytorch' in filename:
        algo = (filename.split("_pytorch")[0]).split("_", 1)[0]
    else:
        raise ValueError('Invalid filename: {logfile}')
    return algo


def get_algo_speeds(logdir=LOG_DIR, framework="jax"):
    logfiles = log_utils.get_logfilenames(LOG_DIR)
    logfiles = [f for f in logfiles if framework in f]

    workloads = sorted(list(set([get_workload_name_from_logfilename(f) for f in logfiles])))
    algos = sorted(list(set([get_algo_name_from_logfilename(f) for f in logfiles])))

    step_time_df = pd.DataFrame(index=algos, columns=workloads)

    total_time = 0 

    for logfile in logfiles:
        workload = get_workload_name_from_logfilename(logfile)
        algo = get_algo_name_from_logfilename(logfile)

        run_df = log_utils.extract_results_df(logfile)
        global_step = run_df['global_step'].iloc[-1]
        total_duration = run_df['total_duration'].iloc[-1]
        total_time = total_time + total_duration
        steps_per_second = global_step / total_duration
        step_time_df.at[algo, workload] = steps_per_second

    print("total time:")
    print(total_time/3600 * 16)
    return step_time_df


df = get_algo_speeds(framework='jax')
print('Jax workload steps/sec:')
df.to_csv("/home/kasimbeg/mlcommons-runs/jax_speed_info.csv")

df = get_algo_speeds(framework='pytorch')
print('Pytorch workload steps/sec:')
df.to_csv("/home/kasimbeg/mlcommons-runs/pytorch_speed_info.csv")
print(df)