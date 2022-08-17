import sys
from datetime import datetime
import gc
import argparse
from libctr import autotuner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='pytorch', help='The model version.')
    parser.add_argument('--config', type=str, default='../config/AFN/initializer_frappe_afn',
                        help='The config file for para tuning.')
    parser.add_argument('--gpu', nargs='+', default=[-1], help='The list of gpu indexes, -1 for cpu.')
    args = vars(parser.parse_args())
    gpu_list = args['gpu']
    version = args['version']
    config_dir = args['config']

    # grid search over parameter space combinations
    autotuner.grid_search(version, config_dir, gpu_list)

