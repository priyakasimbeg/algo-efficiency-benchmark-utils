import log_utils
import os
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = 'logs/step_time_logs_v2'
SAVE_DIR = os.path.join('plots', 'v2')

def get_workload_name_from_logfilename(logfile):
    filename = os.path.basename(logfile)
    if 'jax' in filename:
        workload = (filename.split("_jax")[0]).split("_", 1)[1]
    elif 'pytorch' in filename:
        workload = (filename.split("_pytorch")[0]).split("_", 1)[1]
    return workload

def get_algo_name_from_logfilename(logfile):
    filename = os.path.basename(logfile)
    if 'jax' in filename:
        algo = (filename.split("_jax")[0]).split("_", 1)[0]
    elif 'pytorch' in filename:
        algo = (filename.split("_pytorch")[0]).split("_", 1)[0]
    else:
        raise ValueError('Invalid filename: {logfile}')
    return algo


def plot_algo_speeds(logdir=LOG_DIR, framework="jax"):
    logfiles = log_utils.get_logfilenames(LOG_DIR)
    logfiles = [f for f in logfiles if framework in f]

    workloads = sorted(list(set([get_workload_name_from_logfilename(f) for f in logfiles])))
    algos = sorted(list(set([get_algo_name_from_logfilename(f) for f in logfiles])))

    step_time_df = pd.DataFrame(index=algos, columns=workloads)

    # Build step time df
    for logfile in logfiles:
        workload = get_workload_name_from_logfilename(logfile)
        algo = get_algo_name_from_logfilename(logfile)

        run_df = log_utils.extract_results_df(logfile)
        global_step = run_df['global_step'].iloc[1:].array - run_df['global_step'].iloc[:-1].array
        total_duration = run_df['total_duration'].iloc[1:].array - run_df['total_duration'].iloc[:-1].array
        steps_per_sec = global_step / total_duration
        steps = run_df['global_step'].iloc[1:]
        # step_time_df.at[algo, workload] = steps_per_second
        step_time_df.at[algo, workload] = {'steps': steps, 'steps_per_sec': steps_per_sec}

    # Make plots for each entry in df
    for workload in workloads:
        plt.figure()
        for algo in algos:
            print(f'{algo}_{workload}_{framework}')
            print('-' * 20)
            try: 
                x = step_time_df.at[algo, workload]['steps'][:-1]
                y = step_time_df.at[algo, workload]['steps_per_sec'][:-1]
                plt.plot(x, y, label=f'{algo}')
            except TypeError as e:
                print(f'FAILURE: Can\'t plot {algo}_{workload}_{framework}')
        plt.title(f'{workload}_{framework}')
        plt.ylabel('steps per sec')
        plt.xlabel('step')
        plt.legend()
        plt.savefig(os.path.join(SAVE_DIR, f'steps_per_sec_{workload}_{framework}.png'))

    return step_time_df


df = plot_algo_speeds(framework='jax')
print(df)