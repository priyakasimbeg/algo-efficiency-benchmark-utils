"""
Main script to calculate max times.
"""

import log_utils
import os
import pandas as pd
import re
from absl import flags
from absl import app
from tabulate import tabulate
import glob
import shutil


flags.DEFINE_string('experiment_log_dir',
                    None, 
                    'Path to log dir'
                    ) 
FLAGS = flags.FLAGS

METRICS_MAXIMIZE = ['ssim', 'mean_average_precision', 'accuracy', 'bleu']


logfilename_regex = ('(imagenet_resnet|'
                     'imagenet_vit|'
                     'librispeech_conformer|'
                     'librispeech_deepspeech|'
                     'criteo1tb|'
                     'fastmri|'
                     'ogbg|'
                     'wmt)_(pytorch|jax)(.*).log')


# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

def get_workload_name_from_logfilename(logfile):
    filename = os.path.basename(logfile)
    if re.match(logfilename_regex, filename):
        workload = re.match(logfilename_regex, filename).group(1)
        return workload
    else:
        print(f"no workload name {logfile}")

def get_metric_name(logfile):
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

    return metric_name

def get_best_metrics(logfile):
    metric_name = get_metric_name(logfile)
    run_df = log_utils.extract_results_df(logfile)
    
    validation_metrics = run_df[f'validation/{metric_name}']
    test_metrics = run_df[f'test/{metric_name}']
    train_metrics = run_df[f'train/{metric_name}']

    if metric_name in METRICS_MAXIMIZE:
        best_validation_metric = max(validation_metrics)
        best_test_metric = max(test_metrics)
        best_train_metric = max(train_metrics)
    else:
        best_validation_metric = min(validation_metrics)
        best_test_metric = min(test_metrics)
        best_train_metric = min(train_metrics)
    best_metrics = {
                    f'metric_name': f'{metric_name}',
                    f'train/metric': f'{best_train_metric}',
                    f'validation/metric': f'{best_validation_metric}',
                    f'test/metric': f'{best_test_metric}',
                    }

    return best_metrics


def get_best_metrics_for_all_workloads(experiment_dir):
    logfiles = log_utils.get_logfile_paths(experiment_dir)
    best_metric_dict = {}
    for logfile in logfiles:
        try:
            best_metrics = get_best_metrics(logfile)
            best_metric_dict[logfile] = best_metrics
        except Exception as e:
            continue

    return pd.DataFrame(best_metric_dict).T

def print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))


def main(_):
    source_log_dir = '/home/kasimbeg/mlcommons-runs/fastmri_target_resetting_fixed'
    destination_log_dir = 'logs/target_resetting_fastmri_fixed'

    copy = True
    if copy: 
        files = glob.glob(f"{source_log_dir}/*/*_jax/*.log")
        if not os.path.exists(destination_log_dir):
            os.makedirs(destination_log_dir)
        for f in files:
            shutil.copy(f, destination_log_dir)
    
    df = get_best_metrics_for_all_workloads(destination_log_dir)
    df_sorted = df.sort_values(by='validation/metric')
    print("---- SORTED MEASUREMENTS BY VALIDATION METRIC ----")
    print_df(df_sorted)


    metric_name = df['metric_name'][0]
    if metric_name in METRICS_MAXIMIZE:
        validation_target = df['validation/metric'].median()
        df_top_validation = df.sort_values('validation/metric', ascending=False).head(10)
        test_target = df_top_validation.sort_values('test/metric', ascending=False)['test/metric'][-1]
    else:
        validation_target = df['validation/metric'].median()
        df_top_validation = df.sort_values('validation/metric', ascending=True).head(10)
        test_target = df_top_validation.sort_values('test/metric', ascending=True)['test/metric'][-1]

    
    print(f'VALIDATION TARGET: {validation_target}')
    print(f'TEST TARGET: {test_target}')
    

if __name__ == '__main__':
    app.run(main)
