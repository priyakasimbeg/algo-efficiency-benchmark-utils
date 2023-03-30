import json
import pandas as pd
import os

LOG_FILE = 'logs/fastmri_jax_submission.log'
metrics_prefix = 'Metrics: '

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

def get_logfilenames(logdir):
    filenames = os.listdir(logdir)
    logfilenames = []
    for f in filenames:
        if f.endswith(".log"):
            f = os.path.join(logdir, f)
            if f == 'librispeech_deepspeech_jax_03-15-2023-19-07-09.log':
                continue
            logfilenames.append(f)
    return logfilenames

def get_workload_name_from_logfilename(logfilename):
    if "jax" in logfilename:
        return logfilename.split("_jax")[0]
    elif "pytorch" in logfilename:
        return logfilename.split("_pytorch")[0]

def init_dict_of_lists(keys):
    dict_of_lists = {}
    for key in keys:
        dict_of_lists[key] = []
    return dict_of_lists

def get_metrics_line(logfile):
    with open(logfile, 'r') as f:
        for line in f:
            if metrics_prefix in line:
                return line
    print(f"Log file does not have a metrics line {logfile}")

def convert_metrics_line_to_dict(line):
    eval_results = []
    dict_str = line.split(metrics_prefix)[1]
    dict_str = dict_str.replace("'", "\"")
    dict_str = dict_str.replace("(", "")
    dict_str = dict_str.replace(")", "")
    metrics_dict = json.loads(dict_str)
    for item in metrics_dict['eval_results']:
        if isinstance(item, dict):
            eval_results.append(item)
    
    keys = eval_results[0].keys()
    dict_of_lists = init_dict_of_lists(keys)
    
    for eval_results_dict in eval_results:
        for key in eval_results_dict.keys():
            val = eval_results_dict[key]
            dict_of_lists[key].append(val)
    
    return dict_of_lists

def extract_results_df(logfile):
    line = get_metrics_line(logfile)
    results = convert_metrics_line_to_dict(line)
    return pd.DataFrame(results)

def get_max_time(workload, df):
    end_time = df['total_duration'].iloc[-1]
    print(f"End time: {end_time}")
    end_step = df['global_step'].iloc[-1]
    print(f"End step: {end_step}")
    max_time = end_time/end_step * MAX_STEPS[workload]
    return max_time

def get_max_times_from_logdir(logdir):
    logfilenames = get_logfilenames(logdir)
    for f in logfilenames:
        print(f"Log name: {f}")
        results = extract_results_df(f)
        max_time = get_max_time('fastmri', results)
        print(f"Max time: {max_time}")


get_max_times_from_logdir("logs")
