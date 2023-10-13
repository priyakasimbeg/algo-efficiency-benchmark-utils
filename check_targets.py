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


flags.DEFINE_string('experiment_log_dir',
                    None, 
                    'Path to log dir'
                    ) 
FLAGS = flags.FLAGS


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
    train_metrics = run_df[f'train/{metric_name}']

    if metric_name in ['ssim', 'mean_average_precision', 'accuracy', 'bleu']:
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
            print(f"Could not find metrics line for {logfile}")
            continue

    return pd.DataFrame(best_metric_dict).T

def main(_):
    if not FLAGS.experiment_log_dir:
        log_dir = f'logs/targets_check/pytorch'
    else:
        log_dir = FLAGS.experiment_log_dir
    df = get_best_metrics_for_all_workloads(log_dir)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    # print(df)

if __name__ == '__main__':
    app.run(main)
