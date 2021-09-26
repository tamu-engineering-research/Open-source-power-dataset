# Created by xunannancy at 2021/9/25
"""
difference with svr:
when considering multiple locations:
    multivariate output is possible
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import yaml
from utils import merge_parameters, sklearn_validation, prediction_interval_multiplier, task_prediction_horizon, \
    sklearn_testing, run_evaluate_V3
import numpy as np
import os
from sklearn.model_selection import ParameterGrid
import json

def grid_search_gradient_boosting(config, num_files):
    interval_multiplier = prediction_interval_multiplier[str(config['exp_params']['prediction_interval'])]

    np.random.seed(config['logging_params']['manual_seed'])

    saved_folder = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
    flag = True
    while flag:
        if config['exp_params']['test_flag']:
            last_version = config['exp_params']['last_version'] - 1
        else:
            if not os.path.exists(saved_folder):
                os.makedirs(saved_folder)
                last_version = -1
            else:
                last_version = sorted([int(i.split('_')[1]) for i in os.listdir(saved_folder) if i.startswith('version_')])[-1]
        log_dir = os.path.join(saved_folder, f'version_{last_version+1}')
        if config['exp_params']['test_flag']:
            assert os.path.exists(log_dir)
            flag = False
        else:
            try:
                os.makedirs(log_dir)
                flag = False
            except:
                flag = True
    print(f'log_dir: {log_dir}')

    data_folder = config['exp_params']['data_folder']

    file_list = sorted([i for i in os.listdir(data_folder) if 'zone' in i and i.endswith('.csv')])[:num_files]
    param_grid = {
        'sliding_window': config['exp_params']['sliding_window'],
        'variate': config['exp_params']['variate'],
        'external_feature_flag': config['exp_params']['external_feature_flag'],
        'n_estimators': config['model_params']['n_estimators'],
        'learning_rate': config['model_params']['learning_rate'],
    }
    param_dict_list = list(ParameterGrid(param_grid))

    """
    getting validation results
    """
    if not config['exp_params']['test_flag']:
        print('training...')
        summary = dict()
        for task_index, task_file in enumerate(file_list):
            print(f'{task_index}/{len(file_list)}')
            cur_summary = sklearn_validation(os.path.join(data_folder, task_file), param_dict_list, config, log_dir)
            summary[task_file] = cur_summary
        with open(os.path.join(log_dir, 'param.json'), 'w') as f:
            json.dump(summary, f, indent=4)
    # load params
    param_summary = json.load(open(os.path.join(log_dir, 'param.json'), 'r'))
    """
    prediction on testing
    """
    print('testing...')
    for task_index, task_file in enumerate(file_list):
        print(f'{task_index}/{len(file_list)}')
        predictions = sklearn_testing(os.path.join(data_folder, task_file), param_summary[task_file], config, interval_multiplier, log_dir)
        new_columns = ['ID']
        for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
            for one_horizon in task_prediction_horizon_list:
                print(f'task: {task_name}, horizon: {one_horizon}')
                new_columns += [f'y{task_name[0]}_t+{one_horizon}(mean)', f'y{task_name[0]}_t+{one_horizon}(U)', f'y{task_name[0]}_t+{one_horizon}(L)']
        predictions.to_csv(os.path.join(log_dir, task_file), index=False, columns=new_columns)
    # save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # run evaluate
    evaluate_config = {
        'exp_params': {
            'prediction_path': log_dir,
            'prediction_interval': config['exp_params']['prediction_interval'],
        }
    }
    run_evaluate_V3(config=evaluate_config, verbose=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--manual_seed', '-manual_seed', type=int, help='random seed')
    parser.add_argument('--num_files', '-num_files', type=int, default=3, help='number of files to predict')
    parser.add_argument('--gpus', '-g', type=str)#, default='[3]')

    parser.add_argument('--sliding_window', '-sliding_window', type=str, help='list of sliding_window for arima')
    parser.add_argument('--selection_metric', '-selection_metric', type=str, help='metrics to select hyperparameters, one of [RMSE, MAE, MAPE]',)
    parser.add_argument('--train_valid_ratio', '-train_valid_ratio', type=float, help='select hyperparameters on validation set')
    parser.add_argument('--external_feature_flag', '-external_feature_flag', type=bool, help='whether to consider external features')
    parser.add_argument('--external_features', '-external_features', type=str, help='list of external feature name list')

    # model-specific features
    parser.add_argument('--n_estimators', '-n_estimators', type=str, help='list of n_estimators for grid search')
    parser.add_argument('--learning_rate', '-learning_rate', type=str, help='learning_rate')

    args = vars(parser.parse_args())
    with open('./../configs/gradient_boosting.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    grid_search_gradient_boosting(config, num_files=args['num_files'])

