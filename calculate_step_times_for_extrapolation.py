import log_utils
import os
import pandas as pd

LOG_DIR = 'logs/step_time_logs_v3_a'
OUTPUT_DIR = 'tables/speed_v3_a_redo'
OUTPUT_FILENAME = 'jax_speed_info_3_a.csv'

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


def get_algo_speeds(logdir=LOG_DIR, framework="jax"):

    # Set up dataframe
    logfiles = log_utils.get_logfile_paths(LOG_DIR)
    logfiles = [f for f in logfiles if framework in f]

    workloads = sorted(list(set([get_workload_name_from_logfilename(f) for f in logfiles])))
    algos = sorted(list(set([get_algo_name_from_logfilename(f) for f in logfiles])))

    columns = []
    workload_algo = []
    for workload in workloads:
        for algo in algos:
            workload_algo.append(f'{workload}_{algo}')

    df = pd.DataFrame(index=workload_algo, columns=columns)

    # Populate dataframe
    for logfile in logfiles:
        workload = get_workload_name_from_logfilename(logfile)
        algo = get_algo_name_from_logfilename(logfile)
        index = f'{workload}_{algo}'

        try:
            run_df = log_utils.extract_results_df(logfile)
        except ValueError as e:
            continue
        print(run_df)
        df.at[index, 'max steps'] = MAX_STEPS[workload]
        df.at[index, 'global step'] = run_df['global_step'].iloc[-1]
        df.at[index, 'total duration'] = run_df['total_duration'].iloc[-1]
        df.at[index, 'total eval time'] = run_df['accumulated_eval_time'].iloc[-1]
        df.at[index, 'total logging & checkpointing_time'] = run_df['accumulated_logging_time'].iloc[-1]
        df.at[index, 'total submission time'] = run_df['accumulated_submission_time'].iloc[-1]
        df.at[index, 'submission time till eval 2'] = run_df['accumulated_submission_time'].iloc[1]
        df.at[index, 'num steps till eval 2'] = run_df['global_step'].iloc[1]
        df.at[index, 'submission time since eval 2'] = run_df['accumulated_submission_time'].iloc[-1] - run_df['accumulated_submission_time'].iloc[1]
        df.at[index, 'num steps since eval 2'] = run_df['global_step'].iloc[-1] - run_df['global_step'].iloc[1]

    return df


df = get_algo_speeds(framework='jax')
print(df)
print(df.keys())
df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME))

