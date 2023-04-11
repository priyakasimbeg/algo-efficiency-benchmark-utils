import log_utils
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = 'logs/step_time_test_logs'
SAVE_DIR = os.path.join('plots', 'timing_test')
workload = 'librispeech_conformer'

def get_color_and_alpha_for_run(logfile):
    vm_index = logfile.split('-')[0][0]
    run_index = logfile.split('.log')[0][-1]
    if vm == 1:
        color = 'red'
    if vm == 2:
        color = 'blue'
    if vm == 3:
        color = 'green'
    if vm == 4:
        color = 'yellow'

    if run_index == 1:
        alpha = 0.8
    else:
        alpha = 1
    return color, alpha

def plot_speeds(log_dir='logs/step_time_test_logs',
                plot_first_batch=False,
                save_dir='plots/timing_test'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logfiles = log_utils.get_logfilenames(LOG_DIR)

    step_time_df = pd.DataFrame(index=[os.path.basename(f) for f in logfiles], columns=[workload])
    
    # Build step time df
    for logfile in logfiles:
        print(logfile)
        run_df = log_utils.extract_results_df(logfile)
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

    print(step_time_df)

    plt.figure()
    for logfile in logfiles:
        logfile = os.path.basename(logfile)
        run_name = f'{logfile}'
        print(f'Run name: {run_name}')
        print('-' * 20)
        try: 
            x = step_time_df.at[logfile, workload]['steps'][:-1]
            y = step_time_df.at[logfile, workload]['steps_per_sec'][:-1]
            plt.plot(x, y, label=logfile)
        except TypeError as e:
            print(f'FAILURE: Can\'t plot {logfile}')
    plt.title(f'librispeech_conformer momentum')
    plt.ylabel('steps per sec')
    plt.xlabel('step')
    plt.xticks([x_ for x_ in x ], rotation='vertical')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f'librispeech_conformer.png'))

df = plot_speeds()
print(df)