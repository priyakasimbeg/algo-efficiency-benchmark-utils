import log_utils
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = 'logs/step_time_test_logs'
SAVE_DIR = os.path.join('plots', 'timing_test')
workload = 'librispeech_conformer'

def get_color_and_alpha_for_run(logfile):
    vm = int(logfile.split('-')[1][0])
    run_index = int(logfile.split('.log')[0][-1])
    if vm == 1:
        color = 'red'
    if vm == 2:
        color = 'blue'
    if vm == 3:
        color = 'green'
    if vm == 4:
        color = 'purple'

    if run_index == 1:
        alpha = 0.5
    else:
        alpha = 1
    return color, alpha

def plot_speeds(log_dir='logs/step_time_test_logs',
                plot_first_batch=False,
                save_dir='plots/timing_test'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logfiles = log_utils.get_logfile_paths(LOG_DIR)
    logfiles = sorted(logfiles)

    step_time_df = pd.DataFrame(index=[os.path.basename(f) for f in logfiles], columns=[workload])
    
    run_names = []
    # Build step time df
    for logfile in logfiles:
        print(logfile)
        run_df = log_utils.extract_results_df(logfile)
        print('number of evals: ')
        print(len(run_df['global_step']))
        global_step = run_df['global_step'].iloc[1:].to_numpy() - run_df['global_step'].iloc[:-1].to_numpy()
        total_duration = run_df['total_duration'].iloc[1:].to_numpy()- run_df['total_duration'].iloc[:-1].to_numpy()
        steps_per_sec = global_step / total_duration
        steps = run_df['global_step'].iloc[1:].to_numpy()
        if plot_first_batch == True:
            steps_per_sec_first_batch = np.array([run_df['global_step'].iloc[0] / run_df['total_duration'].iloc[0]])
            step_first_batch = np.array([run_df['global_step'].iloc[0]])
            steps_per_sec = np.hstack((steps_per_sec_first_batch, steps_per_sec))
            steps = np.hstack((step_first_batch, steps))

        step_time_df.at[os.path.basename(logfile), workload] = {'steps': steps, 'steps_per_sec': steps_per_sec}

    speed_stacked = np.array([1, 2, 3])
    plt.figure()
    for logfile in logfiles:
        logfile = os.path.basename(logfile)
        vm_and_run = (logfile.split('-')[1]).split('.log')[0]
        run_name = f'vm_{vm_and_run}'
        run_names.append(run_name)
        print(f'Run name: {run_name}')
        print('-' * 20)
        try: 
            x = step_time_df.at[logfile, workload]['steps'][:-1]
            y = step_time_df.at[logfile, workload]['steps_per_sec'][:-1]
            speed_stacked = np.vstack((speed_stacked, y))
            color, alpha = get_color_and_alpha_for_run(logfile)
            plt.plot(x, y, label=run_name, color=color, alpha=alpha)
        except TypeError as e:
            print(f'FAILURE: Can\'t plot {logfile}')
    plt.title(f'librispeech_conformer momentum')
    plt.ylabel('steps per sec')
    plt.xlabel('step')
    plt.xticks([x_ for x_ in x ], rotation='vertical')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f'librispeech_conformer.png'))

    speed_df = pd.DataFrame(speed_stacked[1:, :], index=run_names)
    speed_df.to_csv(os.path.join(LOG_DIR, 'steps_per_sec.csv'))
    print(speed_df)

plot_speeds()
