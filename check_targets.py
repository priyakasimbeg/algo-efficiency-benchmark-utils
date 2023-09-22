"""
Main script to calculate max times.
"""

import log_utils
import os
import pandas as pd
import re

# FRAMEWORK = 'pytorch'
# LOG_DIR = 'logs/step_time_logs_v4_a_pytorch_redo'
# OUTPUT_DIR = 'tables/timing_pytorch_v4a'
# OUTPUT_FILENAME = 'pytorch_timing_4a_redo.csv'

# LOG_DIR = 'logs/step_time_deepspeech_fixed_3b'
# OUTPUT_DIR = 'tables/timing_v3_jax_fixed_deepspeech'
# OUTPUT_FILENAME = 'jax_timing_deepspeech_fixed_3b.csv'

FRAMEWORK = 'pytorch'
LOG_DIR = 'logs/targets_check/jax'
OUTPUT_DIR = 'tables/targets_check/jax'
OUTPUT_FILENAME = f'targets_check_jax.csv'


logfilename_regex = ('(imagenet_resnet|'
                     'imagenet_vit|'
                     'librispeech_conformer|'
                     'librispeech_deepspeech|'
                     'criteo1tb|'
                     'fastmri|'
                     'ogbg|'
                     'wmt)_(pytorch|jax)(.*).log')


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_workload_name_from_logfilename(logfile):
    filename = os.path.basename(logfile)
    if re.match(logfilename_regex, filename):
        workload = re.match(logfilename_regex, filename).group(1)
        return workload
    else:
        print(f"no workload name {logfile}")

def get_best_metrics(logfile):
    workload_name = get_workload_name_from_logfilename(logfile)
    
    if workload_name == 'criteo1tb':
        metric_name = 'loss'
    elif workload_name == 'fastmri':
        metric_name = 'ssim'
    elif workload_name == 'ogbg':
        metric_name = 'mean_average_precision'
    elif workload_name == 'imagenet_resnet':
        metric_name = 'accuracy'
    elif workload_name == 'imagenet_vit':
        metric_name = 'accuracy'
    elif workload_name == 'librispeech_deepspeech':
        metric_name = 'wer'
    elif workload_name == 'librispeech_conformer':
        metric_name = 'wer'
    elif workload_name == 'wmt':
        metric_name = 'bleu'


    run_df = log_utils.extract_results_df(logfile)
    
    validation_metrics = run_df[f'validation/{metric_name}']
    test_metrics = run_df[f'test/{metric_name}']

    if metric_name in ['ssim', 'mean_average_precision', 'accuracy', 'bleu']:
        best_validation_metric = max(validation_metrics)
        best_test_metric = max(test_metrics)
    else:
        best_validation_metric = min(validation_metrics)
        best_test_metric = min(test_metrics)
    print(f'Logfile: {logfile}')
    print(f'Best validation/{metric_name}: {best_validation_metric}')
    print(f'Best test/{metric_name}: {best_test_metric}')

def get_best_metrics_for_all_workloads(experiment_dir):
    logfiles = log_utils.get_logfile_paths(experiment_dir)

    for logfile in logfiles:
        try:
            get_best_metrics(logfile)
        except Exception as e:
            continue

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
        df.at[index, 'target setting steps'] = MAX_STEPS[workload]
        df.at[index, 'baseline steps'] = MAX_STEPS[workload]/0.75
        df.at[index, 'global step'] = run_df['global_step'].iloc[-1]
        df.at[index, 'total duration (s)'] = run_df['total_duration'].iloc[-1]
        df.at[index, 'total eval time (s)'] = run_df['accumulated_eval_time'].iloc[-1]
        df.at[index, 'total logging & checkpointing_time (s)'] = run_df['accumulated_logging_time'].iloc[-1]
        try:
            df.at[index, 'total data selection time (s)'] = run_df['accumulated_data_selection_time'].iloc[-1]
        except Exception:
            pass
        df.at[index, 'total submission time (s)'] = run_df['accumulated_submission_time'].iloc[-1]
        df.at[index, 'submission time till eval 2 (s)'] = run_df['accumulated_submission_time'].iloc[1]
        df.at[index, 'num steps till eval 2'] = run_df['global_step'].iloc[1]
        df.at[index, 'submission time since eval 2 (s)'] = run_df['accumulated_submission_time'].iloc[-1] - run_df['accumulated_submission_time'].iloc[1]
        df.at[index, 'num steps since eval 2'] = run_df['global_step'].iloc[-1] - run_df['global_step'].iloc[1]
        df.at[index, 'equilibrium step time (s)'] = df.at[index, 'submission time since eval 2 (s)']/df.at[index, 'num steps since eval 2'] 
        df.at[index, 'projected max submission time (s)'] = df.at[index, 'submission time till eval 2 (s)'] + df.at[index, 'equilibrium step time (s)'] * (df.at[index, 'baseline steps'] - df.at[index, 'num steps till eval 2'])

    return df


# df = get_algo_speeds(framework=FRAMEWORK)
# print(df)
# print(df.keys())
# df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME))

# get_best_metrics('logs/targets_check/jax/criteo1tb_jax_09-14-2023-04-33-24.log')
get_best_metrics_for_all_workloads('logs/targets_check/jax_run003')