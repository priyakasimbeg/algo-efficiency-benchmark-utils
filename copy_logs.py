import os

root_dir = "/home/kasimbeg/mlcommons-runs/timing_v2/"
sub_dirs = os.listdir(root_dir)

destination_dir = "/home/kasimbeg/algo-efficiency-timing/logs/v2"

for algo_dir in sub_dirs:
    if 'timing_' in algo_dir:
        algo = algo_dir.split("_")[1]
        algo_path = os.path.join(root_dir, algo_dir)
        workload_dirs = os.listdir(algo_path)
        for workload_dir in workload_dirs:
            workload_path = os.path.join(algo_path, workload_dir)
            workload_contents = os.listdir(workload_path)
            log_files = [f for f in workload_contents if f.endswith(".log")]
            for log_file in log_files:
                log_file_source_path = os.path.join(workload_path, log_file)
                log_file_destination_path = os.path.join(destination_dir, f"{algo}_{log_file}")
                os.system(f"cp {log_file_source_path} {log_file_destination_path}")
