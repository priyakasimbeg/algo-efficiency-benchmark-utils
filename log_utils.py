import os
import pandas as pd
import json

metrics_prefix = 'Metrics: '

def get_logfilenames(logdir):
    filenames = os.listdir(logdir)
    logfilenames = []
    for f in filenames:
        if f.endswith(".log"):
            f = os.path.join(logdir, f)
            logfilenames.append(f)
    return logfilenames


def get_metrics_line(logfile):
    with open(logfile, 'r') as f:
        for line in f:
            if metrics_prefix in line:
                return line
    raise ValueError(f"Log file does not have a metrics line {logfile}")

def convert_metrics_line_to_dict(line):
    eval_results = []
    dict_str = line.split(metrics_prefix)[1]
    dict_str = dict_str.replace("'", "\"")
    dict_str = dict_str.replace("(", "")
    dict_str = dict_str.replace(")", "")
    dict_str = dict_str.replace("DeviceArray", "")
    dict_str = dict_str.replace(", dtype=float32", "")
    dict_str = dict_str.replace("nan", "0")
    try:
        metrics_dict = json.loads(dict_str)
    except Exception as e:
        print(dict_str[830:])
        raise(e)
    for item in metrics_dict['eval_results']:
        if isinstance(item, dict):
            eval_results.append(item)
    
    keys = eval_results[0].keys()

    dict_of_lists = {}
    for key in keys:
        dict_of_lists[key] = []
   
    for eval_results_dict in eval_results:
        for key in eval_results_dict.keys():
            val = eval_results_dict[key]
            dict_of_lists[key].append(val)
    
    return dict_of_lists

def extract_results_df(logfile):
    line = get_metrics_line(logfile)
    results = convert_metrics_line_to_dict(line)
    return pd.DataFrame(results)