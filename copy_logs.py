import os
import log_utils
root_dir = "/home/kasimbeg/mlcommons-runs/timing_fancy"
sub_dirs = os.listdir(root_dir)

destination_dir = "/home/kasimbeg/algo-efficiency-timing/logs/step_time_fancy"

def copy_logs_per_experiment_dir():
    for algo_dir in sub_dirs:
        if 'timing_' in algo_dir:
            algo = algo_dir.split("_")[1]
            algo_path = os.path.join(root_dir, algo_dir)
            workload_dirs = os.listdir(algo_path)
            for workload_dir in workload_dirs:
                workload_path = os.path.join(algo_path, workload_dir)
                try:
                    workload_contents = os.listdir(workload_path)
                except NotADirectoryError as e:
                    continue
                log_files = [f for f in workload_contents if f.endswith(".log")]
                for log_file in log_files:
                    log_file_source_path = os.path.join(workload_path, log_file)
                    try: 
                        _ = log_utils.extract_results_df(log_file_source_path)
                    except Exception as e:
                        print(e)
                        continue
                    log_file_destination_path = os.path.join(destination_dir, f"{algo}_{log_file}")
                    os.system(f"cp {log_file_source_path} {log_file_destination_path}")

def copy_logs_from_experiment_dir():
    for experiment_dir in sub_dirs:
        if 'timing_' in experiment_dir:
            experiment_path = os.path.join(root_dir, experiment_dir)
            experiment_dir_contents = os.listdir(experiment_path)
            log_files = [f for f in experiment_dir_contents if f.endswith(".log")]
            for log_file in log_files:
                log_file_source_path = os.path.join(experiment_path, log_file)
                log_file_destination_path = os.path.join(destination_dir, f'{experiment_dir}')
                os.system(f"cp {log_file_source_path} {log_file_destination_path}")

if not os.path.exists(destination_dir):
    print(f'Making directory {destination_dir}')
    os.makedirs(destination_dir)
copy_logs_per_experiment_dir()
