import os

root_dir = "/home/kasimbeg/mlcommons-runs/"
sub_dirs = os.listdir(root_dir)

destination_dir = "/home/kasimbeg/algo-efficiency-timing/measurements/v1"

if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

for algo_dir in sub_dirs:
    if 'timing_' in algo_dir:
        algo = algo_dir.split("_")[1]
        algo_path = os.path.join(root_dir, algo_dir)
        workload_dirs = os.listdir(algo_path)
        for workload_dir in workload_dirs:
            measurement_source_path = os.path.join(algo_path, workload_dir, 'trial_1', 'measurements.csv')
            measurement_destination_path = os.path.join(destination_dir, f"{algo}_{workload_dir}_measurements.csv")
            # print(f"cp {measurement_source_path} {measurement_destination_path}")
            os.system(f"cp {measurement_source_path} {measurement_destination_path}")
