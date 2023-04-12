import log_utils
import os
import pandas as pd

LOG_DIR = 'logs/step_time_logs_v2'
OUTPUT_DIR = 'tables/speed_v2_per_phase'

MAX_STEPS = {
    'imagenet_resnet': 140000,
    'imagenet_vit': 140000,
    'criteo1tb': 8000,
    'fastmri': 27142,
    'ogbg': 60000,
    'wmt': 100000,
    'librispeech_conformer': 100000,
    'librispeech_deepspeech': 60000,
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

def get_equilibrium_speed(df):
    total_steps = df['global_step'].iloc[-1] - df['global_step'].iloc[0] 
    total_duration = df['total_duration'].iloc[-1] - df['total_duration'].iloc[0]
    return total_steps/total_duration

def get_warmup_speed(df):
    total_steps = df['global_step'].iloc[0]
    total_duration = df['total_duration'].iloc[0]
    return total_steps/total_duration

def get_algo_speeds(logdir=LOG_DIR, framework="jax"):
    logfiles = log_utils.get_logfile_paths(LOG_DIR)
    logfiles = [f for f in logfiles if framework in f]

    workloads = sorted(list(set([get_workload_name_from_logfilename(f) for f in logfiles])))
    algos = sorted(list(set([get_algo_name_from_logfilename(f) for f in logfiles])))

    columns = ['warmup steps/sec', 'equilibrium steps/sec', 'max steps']
    workload_algo = []
    for workload in workloads:
        for algo in algos:
            workload_algo.append(f'{workload}_{algo}')

    step_time_df = pd.DataFrame(index=workload_algo, columns=columns)


    for logfile in logfiles:
        workload = get_workload_name_from_logfilename(logfile)
        algo = get_algo_name_from_logfilename(logfile)

        run_df = log_utils.extract_results_df(logfile)
        step_time_df.at[f'{workload}_{algo}', 'warmup steps/sec'] = get_warmup_speed(run_df)
        step_time_df.at[f'{workload}_{algo}', 'equilibrium steps/sec'] = get_equilibrium_speed(run_df)
        step_time_df.at[f'{workload}_{algo}', 'max steps'] = MAX_STEPS[workload]

    return step_time_df


df = get_algo_speeds(framework='jax')
print('Jax workload steps/sec:')
print(df)
df.to_csv(os.path.join(OUTPUT_DIR, 'jax_speed_info.csv'))

df = get_algo_speeds(framework='pytorch')
print('Pytorch workload steps/sec:')
print(df)
df.to_csv(os.path.join(OUTPUT_DIR, 'pytorch_speed_info.csv'))
