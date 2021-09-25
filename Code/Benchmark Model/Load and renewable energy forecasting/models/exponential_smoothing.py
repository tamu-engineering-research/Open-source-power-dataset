# Created by xunannancy at 2021/9/25
"""
without external features considered & without other target features considered & multiple locations disallowed
three models:
1. simple exponential smoothing: https://otexts.com/fpp2/ses.html
2. Holt’s linear trend method: https://otexts.com/fpp2/holt.html
3. Holt-Winters’ seasonal method: https://otexts.com/fpp2/holt-winters.html
"""
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import numpy as np
import argparse
import yaml
from utils import merge_parameters, task_prediction_horizon, prediction_interval_multiplier, run_evaluate, run_evaluate_V3
import os
import pandas as pd
from sklearn.model_selection import ParameterGrid
import json
from tqdm import tqdm



def exponential_smoothing_validation(file, param_dict_list, config):
    data = pd.read_csv(file)
    train_flag = data['train_flag'].to_numpy()
    training_index = sorted(np.argwhere(train_flag == 1).reshape([-1]))

    summary = dict()
    for param_index, param_dict in tqdm(enumerate(param_dict_list), total=len(param_dict_list)):
        sliding_window, ep_method = param_dict['sliding_window'], param_dict['ep_method']
        results = dict()

        for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
            if task_name == 'wind':
                num_train = int(len(training_index) * config['exp_params']['train_valid_ratio'] * 1.01)
            elif task_name == 'solar':
                num_train = int(len(training_index) * (config['exp_params']['train_valid_ratio'] + 0.005))
            else:
                num_train = int(len(training_index) * config['exp_params']['train_valid_ratio'])

            total_y_t = data[f'y{task_name[0]}_t'].to_numpy()
            for horizon in task_prediction_horizon_list:
                results[f'y{task_name[0]}_t+{horizon}'] = [[], []]
            for val_index in range(len(training_index) - num_train):
                print(f'uni-{task_name}, {val_index}/{len(training_index) - num_train}')
                flag_prod = 0
                for horizon in task_prediction_horizon_list:
                    flag_prod += data[f'y{task_name[0]}_t+{horizon}(flag)'][num_train+val_index]
                if flag_prod == 0:
                    continue
                history_y_t = total_y_t[num_train+val_index+1-sliding_window:num_train+val_index+1]
                if ep_method == 'simple':
                    model_fit = SimpleExpSmoothing(history_y_t, initialization_method="estimated").fit()
                elif ep_method == 'Holt':
                    model_fit = Holt(history_y_t, damped_trend=True, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2,)
                elif ep_method == 'Holt Winters':
                    model_fit = ExponentialSmoothing(history_y_t, seasonal_periods=4, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
                predictions = model_fit.forecast(
                    steps=int(np.max(task_prediction_horizon_list))
                )
                for horizon in task_prediction_horizon_list:
                    if data[f'y{task_name[0]}_t+{horizon}(flag)'][num_train+val_index] == 1:
                        results[f'y{task_name[0]}_t+{horizon}'][0].append(data[f'y{task_name[0]}_t+{horizon}(val)'][num_train+val_index])
                        results[f'y{task_name[0]}_t+{horizon}'][1].append(predictions[horizon-1])
        cur_summary = run_evaluate(results, config['exp_params']['selection_metric'])
        for task_horizon, res in cur_summary.items():
            if task_horizon not in summary:
                summary[task_horizon] = {
                    'param_dict': [param_dict],
                    'results': [res],
                }
            else:
                summary[task_horizon]['param_dict'].append(param_dict)
                summary[task_horizon]['results'].append(res)
    # hyperparameter selection
    selected_summary = dict()
    for task_horizon, param_res in summary.items():
        selected_index = np.argmin(np.array(param_res['results'])[:, 0])
        selected_summary[task_horizon] = {
            'param_dict': param_res['param_dict'][selected_index],
            'mean': param_res['results'][selected_index][0],
            'std': param_res['results'][selected_index][1]
        }
    return selected_summary

def exponential_smoothing_testing(file, param_summary, config, interval_multiplier):
    data = pd.read_csv(file)
    train_flag = data['train_flag'].to_numpy()
    testing_index = sorted(np.argwhere(train_flag == 0).reshape([-1]))
    testing_data = data.iloc[testing_index]
    testing_ID = testing_data['ID'].to_numpy()

    testing_predictions, columns = list(), list()
    for idx, (task_horizon, res) in enumerate(param_summary.items()):
        print(f'task_horizon: {task_horizon}')
        param_dict, std = res['param_dict'], res['std']
        sliding_window, ep_method = param_dict['sliding_window'], param_dict['ep_method']

        task_name, horizon_val = task_horizon.split('_')[0], int(task_horizon[task_horizon.find('+')+1:])
        task_name = {
            'yl': 'load',
            'yw': 'wind',
            'ys': 'solar'
        }[task_name]
        total_y_t = data[f'y{task_name[0]}_t'].to_numpy()
        cur_predictions = -1 * np.ones([len(testing_data)])
        for iter, index in enumerate(testing_index):
            if data[f'y{task_name[0]}_t+{horizon_val}(flag)'][index] == 0:
                continue
            # print(f'{iter}/{len(testing_index)}')
            history_y_t = total_y_t[index+1-sliding_window:index+1]
            if ep_method == 'simple':
                model_fit = SimpleExpSmoothing(history_y_t, initialization_method="estimated").fit()
            elif ep_method == 'Holt':
                model_fit = Holt(history_y_t, damped_trend=True, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2,)
            elif ep_method == 'Holt Winters':
                model_fit = ExponentialSmoothing(history_y_t, seasonal_periods=4, trend='add', seasonal='add', damped_trend=True, use_boxcox=True, initialization_method="estimated").fit()
            prediction = model_fit.forecast(
                steps=horizon_val
            )[-1]
            cur_predictions[iter] = prediction
        testing_predictions.append(cur_predictions)
        columns.append(f'{task_horizon}(mean)')
        # if idx == 1:
        #     break
    testing_predictions = np.stack(testing_predictions, axis=-1)
    predictions = pd.DataFrame(
        data=testing_predictions,
        columns=columns,
        index=data.index[testing_index].to_numpy()
    )
    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        for one_horizon in task_prediction_horizon_list:
            std = param_summary[f'y{task_name[0]}_t+{one_horizon}']['std']
            predictions[f'y{task_name[0]}_t+{one_horizon}(U)'] = predictions[f'y{task_name[0]}_t+{one_horizon}(mean)'] + interval_multiplier * std
            predictions[f'y{task_name[0]}_t+{one_horizon}(L)'] = predictions[f'y{task_name[0]}_t+{one_horizon}(mean)'] - interval_multiplier * std
    predictions['ID'] = testing_ID
    return predictions

def grid_search_ep(config, num_files):
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

    # file_list = sorted([i for i in os.listdir(data_folder) if 'zone' in i and i.endswith('.csv')])[:num_files]
    file_list = sorted([i for i in os.listdir(data_folder) if 'zone' in i and i.endswith('.csv')])[1:3]
    param_grid = {
        'sliding_window': config['exp_params']['sliding_window'],
        'ep_method': ['simple', 'Holt'], #, 'Holt Winters']
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
            cur_summary = exponential_smoothing_validation(os.path.join(data_folder, task_file), param_dict_list, config)
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
        predictions = exponential_smoothing_testing(os.path.join(data_folder, task_file), param_summary[task_file], config, interval_multiplier)
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
    parser.add_argument('--num_files', '-num_files', type=int, default=1, help='number of files to predict')
    parser.add_argument('--gpus', '-g', type=str)#, default='[1]')

    parser.add_argument('--sliding_window', '-sliding_window', type=str, help='list of sliding_window for arima')
    parser.add_argument('--selection_metric', '-selection_metric', type=str, help='metrics to select hyperparameters, one of [RMSE, MAE, MAPE]',)
    parser.add_argument('--train_valid_ratio', '-train_valid_ratio', type=float, help='select hyperparameters on validation set')

    args = vars(parser.parse_args())
    with open('./../configs/exponential_smoothing.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    grid_search_ep(config, num_files=args['num_files'])
