import log_utils

logfile = '/home/kasimbeg/mlcommons-runs/timing_test_v3/timing_momentum_kasimbeg-1_full_run0/ogbg_jax/ogbg_jax_04-18-2023-00-43-26.log'

df = log_utils.extract_results_df(logfile)
print(df.to_csv())