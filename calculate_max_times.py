import json
import pandas as pd
import os
import log_utils


MAX_STEPS = {
    'imagenet_resnet': 140000,
    'imagenet_vit': 140000,
    'criteo': 8000,
    'fastmri': 27142,
    'ogbg': 60000,
    'wmt': 100000,
    'librispeech_conformer': 100000,
    'librispeech_deepspeech': 60000,
}

def get_workload_name_from_logfilename(logfilename):
    if "jax" in logfilename:
        return logfilename.split("_jax")[0]
    elif "pytorch" in logfilename:
        return logfilename.split("_pytorch")[0]

        
def get_max_time(workload, df):
    end_time = df['total_duration'].iloc[-1]
    print(f"End time: {end_time}")
    end_step = df['global_step'].iloc[-1]
    print(f"End step: {end_step}")
    max_time = end_time/end_step * MAX_STEPS[workload]
    return max_time

def get_max_times_from_logdir(logdir):
    logfilenames =log_utils.get_logfilenames(logdir)
    for f in logfilenames:
        print(f"Log name: {f}")
        results = log_utils.extract_results_df(f)
        max_time = get_max_time('fastmri', results)
        print(f"Max time: {max_time}")


get_max_times_from_logdir("logs")
