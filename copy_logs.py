import os
import log_utils

root_dir = "/home/kasimbeg/mlcommons-runs/timing_pytorch_nightly_2023_08_20"
destination_dir = "/home/kasimbeg/algo-efficiency-timing/logs/timing_pytorch_nightly_2023_08_20"
# root_dir = "/home/kasimbeg/mlcommons-runs/timing_fancy_jax_upgrade_b"
# destination_dir = "/home/kasimbeg/algo-efficiency-timing/logs/step_time_fancy_deepspeech_fixed_b"

def copy_logs_per_experiment_dir(root_dir=root_dir, logfile_filter_string=None):
    sub_dirs = os.listdir(root_dir)
    for algo_dir in sub_dirs:
        if 'timing_' in algo_dir:
            algo = algo_dir.split("_")[1]
        else:
            algo = algo_dir
        algo_path = os.path.join(root_dir, algo_dir)
        workload_dirs = os.listdir(algo_path)
        for workload_dir in workload_dirs:
            workload_path = os.path.join(algo_path, workload_dir)
            try:
                workload_contents = os.listdir(workload_path)
            except NotADirectoryError as e:
                continue
            log_files = [f for f in workload_contents if f.endswith(".log")]
            print(log_files)
            for log_file in log_files:
                log_file_source_path = os.path.join(workload_path, log_file)
                try: 
                    _ = log_utils.extract_results_df(log_file_source_path)
                except Exception as e:
                    print(e)
                    print(log_file)
                    continue
                log_file_destination_path = os.path.join(destination_dir, f"{algo}_{log_file}")
                if logfile_filter_string:
                    print(log_file_source_path)
                    filter = logfile_filter_string in log_file_source_path
                    print(filter)
                    if logfile_filter_string in log_file_source_path:
                        print(log_file_destination_path)
                        os.system(f"cp {log_file_source_path} {log_file_destination_path}")
                else:
                    print(log_file_destination_path)
                    os.system(f"cp {log_file_source_path} {log_file_destination_path}")
 

if not os.path.exists(destination_dir):
    print(f'Making directory {destination_dir}')
    os.makedirs(destination_dir)
    

copy_logs_per_experiment_dir(logfile_filter_string=None)
