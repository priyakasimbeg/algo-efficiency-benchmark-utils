from absl import flags
from absl import app
import os
import docker
import time 

flags.DEFINE_string('algorithm', None,
                    'Optimization algorithm in reference algorithms.')
flags.DEFINE_string('framework', None, 'Can be either pytorch or jax')
flags.DEFINE_boolean('dry_run', False, 'Whether or not to actually run the command')
flags.DEFINE_string('docker_tag', None, 'Optional Docker image tag')
flags.DEFINE_integer('num_runs', 1, 'Number of times to run the container.')
flags.DEFINE_string('workload', None, 'Workload to run.')
flags.DEFINE_string('experiment_postfix', None, 'Postfix on experiment name.')
FLAGS = flags.FLAGS

RUN_FRACTION = 0.1

DATASETS = ['imagenet',
            'fastmri',
            'ogbg',
            'wmt',
            'librispeech',
            'criteo1tb']

WORKLOADS = ['imagenet_resnet'
             'imagenet_vit',
             'fastmri'
             'ogbg',
             'wmt',
             'librispeech_deepspeech',
             'librispeech_conformer',
             'criteo1tb']

WORKLOADS = {'imagenet_resnet': {'max_steps': 140000,
                                 'dataset': 'imagenet'},
             'imagenet_vit': {'max_steps': 140000,
                              'dataset': 'imagenet'},
             'fastmri': {'max_steps': 27142,
                         'dataset': 'fastmri'},
             'ogbg': {'max_steps': 60000,
                      'dataset': 'ogbg'},
             'wmt': {'max_steps': 100000,
                     'dataset': 'wmt'},
             'librispeech_deepspeech': {'max_steps': 80000,
                                        'dataset': 'librispeech'},
             'librispeech_conformer': {'max_steps': 100000,
                                       'dataset': 'librispeech'},
             'criteo1tb': {'max_steps': 8000,
                           'dataset': 'criteo1tb'},
             }

def container_running():
    docker_client = docker.from_env()
    containers = docker_client.containers.list()
    if len(containers) == 0:
        return False
    else:
        return True

def wait_until_container_not_running(sleep_interval=5*60):
    while container_running():
        time.sleep(sleep_interval)
    return 
    
def main(_):
    framework = FLAGS.framework
    algorithm = FLAGS.algorithm
    workload = FLAGS.workload
    docker_tag = f'{FLAGS.docker_tag}' if FLAGS.docker_tag is not None else 'latest'
    num_runs = FLAGS.num_runs
    experiment_postfix = f'_{FLAGS.experiment_postfix}' if FLAGS.experiment_postfix else ''

    if workload not in WORKLOADS:
        raise ValueError(f'Invalid value for workload: {workload}.')

    # For each runnable workload check if there are any containers running and if not launch next container command
    for run in range(num_runs):
        wait_until_container_not_running()
        print('='*100)
        dataset = WORKLOADS[workload]['dataset']
        max_steps = int(WORKLOADS[workload]['max_steps'] * RUN_FRACTION)
        experiment_name = f'timing_test_v2/timing_{algorithm}{experiment_postfix}_run{run}'
        command = ('docker run -t -d -v /home/kasimbeg/data/:/data/ '
                   '-v /home/kasimbeg/experiment_runs/:/experiment_runs '
                   '-v /home/kasimbeg/experiment_runs/logs:/logs '
                   '--gpus all --ipc=host '
                   f'us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/base_image:{docker_tag} '
                   f'-d {dataset} '
                   f'-f {framework} '
                   f'-s baselines/{algorithm}/{framework}/submission.py '
                   f'-w {workload} '
                   f'-t baselines/{algorithm}/tuning_search_space.json '
                   f'-e {experiment_name} '
                   f'-m {max_steps}')
        if not FLAGS.dry_run:
            print('Running docker container command')
            print('Container ID: ')
            return_code = os.system(command)
        else:
            return_code = 0
        if return_code == 0:
            print(f'SUCCESS: container for {framework} {workload} {algorithm} launched successfully')
            print(f'Command: {command}')
            print(f'Results will be logged to {experiment_name}')
        else:
            print(f'Failed: container for {framework} {workload} {algorithm} failed with exit code {return_code}.')
            print(f'Command: {command}')
        print('='*100)
        wait_until_container_not_running()
        if run >= (num_runs/2 - 1):
            os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'") # clear caches



if __name__ == '__main__':
    flags.mark_flag_as_required('framework')
    flags.mark_flag_as_required('algorithm')
    flags.mark_flag_as_required('workload')

    app.run(main)