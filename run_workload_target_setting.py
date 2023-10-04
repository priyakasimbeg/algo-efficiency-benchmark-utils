"""
Example Usage:
python run_workload_target_setting.py \
--framework jax \
--experiment_basename jax_upgrade \
--docker_image_url <url_for_docker_image> \
--tag <some_docker_tag> \
--num_runs 20  
"""

from absl import flags
from absl import app
import os
import docker
import time 


flags.DEFINE_string('framework', None, 'Can be either pytorch or jax')
flags.DEFINE_boolean('dry_run', False, 'Whether or not to actually run the command')
flags.DEFINE_string('tag', None, 'Optional Docker image tag')
flags.DEFINE_string('docker_image_url', 'us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/algoperf_jax_dev', 'URL to docker image') 
flags.DEFINE_string('experiment_basename', 'timing', 'Name of top sub directory in experiment dir.')
flags.DEFINE_boolean('rsync_data', True, 'Whether or not to transfer the data from GCP w rsync.')
flags.DEFINE_integer('num_runs', 1, 'Number of times to repeat a run.')
flags.DEFINE_string('workload', None, 'Workload to run, if None, run all workloads.')
flags.DEFINE_boolean('local', True, 'Whether or not to mount the local algorithmic-efficiency repo to the container.')
FLAGS = flags.FLAGS


DATASETS = ['imagenet',
            'fastmri',
            'ogbg',
            'wmt',
            'librispeech',
            'criteo1tb']

WORKLOAD_NAMES = ['imagenet_resnet',
             'imagenet_vit',
             'fastmri',
             'ogbg',
             'wmt',
             'librispeech_deepspeech',
             'librispeech_conformer',
             'criteo1tb'
             ]

WORKLOADS = {
             'fastmri': {'max_steps': 27142,
                         'dataset': 'fastmri',
                         'algorithm': 'nesterov'},
             'ogbg': {'max_steps': 60000,
                      'dataset': 'ogbg',
                      'algorithm': 'nesterov'},
             'criteo1tb': {'max_steps': 8000,
                           'dataset': 'criteo1tb',
                           'algorithm': 'nadamw'},
             'imagenet_resnet': {'max_steps': 140000,
                                 'dataset': 'imagenet',
                                 'algorithm': 'momentum'},
             'imagenet_vit': {'max_steps': 140000,
                              'dataset': 'imagenet',
                              'algorithm': 'nadamw'},
             'wmt': {'max_steps': 100000,
                     'dataset': 'wmt',
                     'algorithm': 'nadamw'},
             'librispeech_deepspeech': {'max_steps': 36000,
                                        'dataset': 'librispeech',
                                        'algorithm': 'nadamw'},
             'librispeech_conformer': {'max_steps': 60000,
                                       'dataset': 'librispeech',
                                       'algorithm': 'adamw'},
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
    tag = f':{FLAGS.tag}' if FLAGS.tag is not None else ''
    num_runs = FLAGS.num_runs
    experiment_basename=FLAGS.experiment_basename
    rsync_data = 'true' if FLAGS.rsync_data else 'false'
    docker_image_url = FLAGS.docker_image_url
    mount_repo_flag = ''
    if FLAGS.local:
        mount_repo_flag = '-v /home/kasimbeg/algorithmic-efficiency:/algorithmic-efficiency '
    if FLAGS.workload:
        workloads = [FLAGS.workload]
    else:
        workloads = WORKLOAD_NAMES

    # For each runnable workload check if there are any containers running and if not launch next container command
    for workload in workloads:
        for n in range(num_runs):
            wait_until_container_not_running()
            os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'") # clear caches
            print('='*100)
            dataset = WORKLOADS[workload]['dataset']
            max_steps = int(WORKLOADS[workload]['max_steps'])
            algorithm = WORKLOADS[workload]['algorithm']

            experiment_name = f'{experiment_basename}/{algorithm}'
            if workload == 'conformer':
                tuning_tag = '_conformer'
            else:
                tuning_tag = ''
            command = ('docker run -t -d -v /home/kasimbeg/data/:/data/ '
                    '-v /home/kasimbeg/experiment_runs/:/experiment_runs '
                    '-v /home/kasimbeg/experiment_runs/logs:/logs '
                    f'{mount_repo_flag}'
                    '--gpus all --ipc=host '
                    f'{docker_image_url}{tag} '
                    f'-d {dataset} '
                    f'-f {framework} '
                    f'-s reference_algorithms/target_setting_algorithms/{framework}_{algorithm}.py '
                    f'-w {workload} '
                    f'-t reference_algorithms/target_setting_algorithms/{workload}/tuning_search_space.json '
                    f'-e {experiment_name}_run_{n} '
                    f'-m {max_steps} '
                    '-c false '
                    '-o true ' 
                    f'-r {rsync_data} '
                    '-i true ')
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
            wait_until_container_not_running()
            os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'") # clear caches

            print('='*100)


if __name__ == '__main__':
    flags.mark_flag_as_required('framework')

    app.run(main)"""
Example Usage:
python run_workload_target_setting.py \
--framework jax \
--experiment_basename jax_upgrade \
--docker_image_url <url_for_docker_image> \
--tag <some_docker_tag> \
--num_runs 20  
"""

from absl import flags
from absl import app
import os
import docker
import time 


flags.DEFINE_string('framework', None, 'Can be either pytorch or jax')
flags.DEFINE_boolean('dry_run', False, 'Whether or not to actually run the command')
flags.DEFINE_string('tag', None, 'Optional Docker image tag')
flags.DEFINE_string('docker_image_url', 'us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/algoperf_jax_dev', 'URL to docker image') 
flags.DEFINE_string('experiment_basename', 'timing', 'Name of top sub directory in experiment dir.')
flags.DEFINE_boolean('rsync_data', True, 'Whether or not to transfer the data from GCP w rsync.')
flags.DEFINE_integer('num_runs', 1, 'Number of times to repeat a run.')
flags.DEFINE_string('workload', None, 'Workload to run, if None, run all workloads.')
flags.DEFINE_boolean('local', True, 'Whether or not to mount the local algorithmic-efficiency repo to the container.')
FLAGS = flags.FLAGS


DATASETS = ['imagenet',
            'fastmri',
            'ogbg',
            'wmt',
            'librispeech',
            'criteo1tb']

WORKLOAD_NAMES = ['imagenet_resnet',
             'imagenet_vit',
             'fastmri',
             'ogbg',
             'wmt',
             'librispeech_deepspeech',
             'librispeech_conformer',
             'criteo1tb'
             ]

WORKLOADS = {
             'fastmri': {'max_steps': 27142,
                         'dataset': 'fastmri',
                         'algorithm': 'nesterov'},
             'ogbg': {'max_steps': 60000,
                      'dataset': 'ogbg',
                      'algorithm': 'nesterov'},
             'criteo1tb': {'max_steps': 8000,
                           'dataset': 'criteo1tb',
                           'algorithm': 'nadamw'},
             'imagenet_resnet': {'max_steps': 140000,
                                 'dataset': 'imagenet',
                                 'algorithm': 'momentum'},
             'imagenet_vit': {'max_steps': 140000,
                              'dataset': 'imagenet',
                              'algorithm': 'nadamw'},
             'wmt': {'max_steps': 100000,
                     'dataset': 'wmt',
                     'algorithm': 'nadamw'},
             'librispeech_deepspeech': {'max_steps': 36000,
                                        'dataset': 'librispeech',
                                        'algorithm': 'nadamw'},
             'librispeech_conformer': {'max_steps': 60000,
                                       'dataset': 'librispeech',
                                       'algorithm': 'adamw'},
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
    tag = f':{FLAGS.tag}' if FLAGS.tag is not None else ''
    num_runs = FLAGS.num_runs
    experiment_basename=FLAGS.experiment_basename
    rsync_data = 'true' if FLAGS.rsync_data else 'false'
    docker_image_url = FLAGS.docker_image_url
    mount_repo_flag = ''
    if FLAGS.local:
        mount_repo_flag = '-v /home/kasimbeg/algorithmic-efficiency:/algorithmic-efficiency '
    if FLAGS.workload:
        workloads = [FLAGS.workload]
    else:
        workloads = WORKLOAD_NAMES

    # For each runnable workload check if there are any containers running and if not launch next container command
    for workload in workloads:
        for n in range(num_runs):
            wait_until_container_not_running()
            os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'") # clear caches
            print('='*100)
            dataset = WORKLOADS[workload]['dataset']
            max_steps = int(WORKLOADS[workload]['max_steps'])
            algorithm = WORKLOADS[workload]['algorithm']

            experiment_name = f'{experiment_basename}/{algorithm}'
            if workload == 'conformer':
                tuning_tag = '_conformer'
            else:
                tuning_tag = ''
            command = ('docker run -t -d -v /home/kasimbeg/data/:/data/ '
                    '-v /home/kasimbeg/experiment_runs/:/experiment_runs '
                    '-v /home/kasimbeg/experiment_runs/logs:/logs '
                    f'{mount_repo_flag}'
                    '--gpus all --ipc=host '
                    f'{docker_image_url}{tag} '
                    f'-d {dataset} '
                    f'-f {framework} '
                    f'-s reference_algorithms/target_setting_algorithms/{framework}_{algorithm}.py '
                    f'-w {workload} '
                    f'-t reference_algorithms/target_setting_algorithms/{workload}/tuning_search_space.json '
                    f'-e {experiment_name}_run_{n} '
                    f'-m {max_steps} '
                    '-c false '
                    '-o true ' 
                    f'-r {rsync_data} '
                    '-i true ')
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
            wait_until_container_not_running()
            os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'") # clear caches

            print('='*100)


if __name__ == '__main__':
    flags.mark_flag_as_required('framework')

    app.run(main)
