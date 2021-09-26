# Created by xunannancy at 2021/9/25
"""
arima (arma included when d=0)
grid search to find the best hyper-parameters
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import yaml
from utils import merge_parameters, prediction_interval_multiplier, task_prediction_horizon, run_evaluate, run_evaluate_V3
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.arima.model import ARIMA
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
import json
from statsmodels.tsa.vector_ar.var_model import VAR
import itertools
from collections import OrderedDict

def arima_validation(file, param_dict_list, config):
    data = pd.read_csv(file)
    train_flag = data['train_flag'].to_numpy()
    training_index = sorted(np.argwhere(train_flag == 1).reshape([-1]))

    summary = dict()
    for param_index, param_dict in tqdm(enumerate(param_dict_list), total=len(param_dict_list)):
        sliding_window, p = param_dict['sliding_window'], param_dict['p']
        if p == 0:
            continue
        if param_dict['variate_options']['variate'] == 'uni':
            d, q = param_dict['variate_options']['d'], param_dict['variate_options']['q']
            if d == 0 and q == 0:
                continue
            cur_summary = uni_arima_validation(data, training_index, sliding_window, p, d, q, param_dict, config)
        elif param_dict['variate_options']['variate'] == 'multi':
            cur_summary = multi_arima_validation(data, training_index, sliding_window, p, param_dict, config)
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

def uni_arima_validation(data, training_index, sliding_window, p, d, q, param_dict, config):
    results = dict()
    if param_dict['external_feature_flag']:
        total_features = data[config['exp_params']['external_features']].to_numpy()

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
            if param_dict['external_feature_flag']:
                history_features = total_features[num_train+val_index+1-sliding_window:num_train+val_index+1]
                future_features = total_features[num_train+val_index+1:num_train+val_index+1+np.max(task_prediction_horizon_list)]
                model = ARIMA(endog=history_y_t, exog=history_features, order=(p, d, q))
                try:
                    model_fit = model.fit()
                    predictions = model_fit.forecast(
                        steps=int(np.max(task_prediction_horizon_list)),
                        exog=future_features
                    )
                except np.linalg.LinAlgError:
                    print('np.linalg.LinAlgError: Schur decomposition solver error.')
                    predictions = np.repeat([history_y_t[-1]], max(task_prediction_horizon_list))
            else:
                model = ARIMA(endog=history_y_t, order=(p, d, q))
                try:
                    model_fit = model.fit()
                    predictions = model_fit.forecast(
                        steps=int(np.max(task_prediction_horizon_list))
                    )
                except np.linalg.LinAlgError:
                    print('np.linalg.LinAlgError: Schur decomposition solver error.')
                    predictions = np.repeat([history_y_t[-1]], max(task_prediction_horizon_list))
            # NOTE: replace nan with naive
            predictions[np.isnan(predictions)] = history_y_t[-1]
            for horizon in task_prediction_horizon_list:
                if data[f'y{task_name[0]}_t+{horizon}(flag)'][num_train+val_index] == 1:
                    results[f'y{task_name[0]}_t+{horizon}'][0].append(data[f'y{task_name[0]}_t+{horizon}(val)'][num_train+val_index])
                    results[f'y{task_name[0]}_t+{horizon}'][1].append(predictions[horizon-1])
    summary = run_evaluate(results, config['exp_params']['selection_metric'])
    return summary


def multi_arima_validation(data, training_index, sliding_window, p, param_dict, config):
    num_train = int(len(training_index) * config['exp_params']['train_valid_ratio'])

    results = dict()
    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        for horizon in task_prediction_horizon_list:
            results[f'y{task_name[0]}_t+{horizon}'] = [[], []]

    total_y_t = data[[f'y{task_name[0]}_t' for task_name in task_prediction_horizon.keys()]].to_numpy()
    if param_dict['external_feature_flag']:
        total_features = data[config['exp_params']['external_features']].to_numpy()

    for val_index in range(len(training_index) - num_train):
        print(f'multi, {val_index}/{len(training_index) - num_train}')
        flag_prod = 0
        valid_task_prediction_horizon = dict()
        for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
            for horizon in task_prediction_horizon_list:
                cur_flag = data[f'y{task_name[0]}_t+{horizon}(flag)'][num_train+val_index]
                flag_prod += cur_flag
                if cur_flag == 1:
                    if task_name not in valid_task_prediction_horizon:
                        valid_task_prediction_horizon[task_name] = [horizon]
                    else:
                        valid_task_prediction_horizon[task_name].append(horizon)
        if flag_prod == 0:
            continue
        merged_prediction_horizon_list = list(itertools.chain(*valid_task_prediction_horizon.values()))

        history_y_t = total_y_t[num_train+val_index+1-sliding_window:num_train+val_index+1]
        lag_y = total_y_t[num_train+val_index+1-p:num_train+val_index+1]
        if param_dict['external_feature_flag']:
            history_features = total_features[num_train+val_index+1-sliding_window:num_train+val_index+1]
            future_features = total_features[num_train+val_index+1:num_train+val_index+1+np.max(merged_prediction_horizon_list)]
            model = VAR(endog=history_y_t, exog=history_features)
            model_fit = model.fit(p)
            predictions = model_fit.forecast(y=lag_y, steps=int(np.max(merged_prediction_horizon_list)), exog_future=future_features)
        else:
            model = VAR(endog=history_y_t)
            model_fit = model.fit(p)
            predictions = model_fit.forecast(y=lag_y, steps=int(np.max(merged_prediction_horizon_list)))
        for task_name, task_prediction_horizon_list in valid_task_prediction_horizon.items():
            for horizon in task_prediction_horizon_list:
                results[f'y{task_name[0]}_t+{horizon}'][0].append(data[f'y{task_name[0]}_t+{horizon}(val)'][num_train+val_index])
                results[f'y{task_name[0]}_t+{horizon}'][1].append(predictions[horizon-1][list(task_prediction_horizon.keys()).index(task_name)])
    summary = run_evaluate(results, config['exp_params']['selection_metric'])
    return summary

def arima_testing(file, param_summary, config, interval_multiplier):
    data = pd.read_csv(file)
    train_flag = data['train_flag'].to_numpy()
    testing_index = sorted(np.argwhere(train_flag == 0).reshape([-1]))
    testing_data = data.iloc[testing_index]
    testing_ID = testing_data['ID'].to_numpy()

    testing_predictions, columns = list(), list()
    for idx, (task_horizon, res) in enumerate(param_summary.items()):
        print(f'task_horizon: {task_horizon}')
        param_dict, std = res['param_dict'], res['std']
        sliding_window, p = param_dict['sliding_window'], param_dict['p']
        if param_dict['variate_options']['variate'] == 'uni':
            d, q = param_dict['variate_options']['d'], param_dict['variate_options']['q']
            cur_predictions = uni_arima_testing(task_horizon, data, testing_index, sliding_window, p, d, q, param_dict, config)
        elif param_dict['variate_options']['variate'] == 'multi':
            cur_predictions = multi_arima_testing(task_horizon, data, testing_index, sliding_window, p, param_dict, config)
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

def uni_arima_testing(task_horizon, data, testing_index, sliding_window, p, d, q, param_dict, config):
    task_name, horizon_val = task_horizon.split('_')[0], int(task_horizon[task_horizon.find('+')+1:])
    task_name = {
        'yl': 'load',
        'yw': 'wind',
        'ys': 'solar'
    }[task_name]
    testing_data = data.iloc[testing_index]

    if param_dict['external_feature_flag']:
        total_features = data[config['exp_params']['external_features']].to_numpy()

    total_y_t = data[f'y{task_name[0]}_t'].to_numpy()
    testing_predictions = -1 * np.ones([len(testing_data)])
    for iter, index in enumerate(testing_index):
        if data[f'y{task_name[0]}_t+{horizon_val}(flag)'][index] == 0:
            continue
        # print(f'{iter}/{len(testing_index)}')
        history_y_t = total_y_t[index+1-sliding_window:index+1]
        if param_dict['external_feature_flag']:
            history_features = total_features[index+1-sliding_window:index+1]
            future_features = total_features[index+1:index+1+horizon_val]
            model = ARIMA(endog=history_y_t, exog=history_features, order=(p, d, q))
            try:
                model_fit = model.fit()
                prediction = model_fit.forecast(
                    steps=horizon_val,
                    exog=future_features
                )[-1]
            except np.linalg.LinAlgError:
                print('numpy.linalg.LinAlgError: Schur decomposition solver error.')
                prediction = history_y_t[-1]
        else:
            model = ARIMA(endog=history_y_t, order=(p, d, q))
            try:
                model_fit = model.fit()
                prediction = model_fit.forecast(
                    steps=horizon_val
                )[-1]
            except np.linalg.LinAlgError:
                print('numpy.linalg.LinAlgError: Schur decomposition solver error.')
                prediction = history_y_t[-1]
        if np.isnan(prediction):
            prediction = history_y_t[-1]
        testing_predictions[iter] = prediction
    return testing_predictions

def multi_arima_testing(task_horizon, data, testing_index, sliding_window, p, param_dict, config):
    task_name, horizon_val = task_horizon.split('_')[0], int(task_horizon[task_horizon.find('+')+1:])
    task_name = {
        'yl': 'load',
        'yw': 'wind',
        'ys': 'solar'
    }[task_name]
    total_y_t = data[[f'y{l[0]}_t' for l in task_prediction_horizon.keys()]].to_numpy()
    if param_dict['external_feature_flag']:
        total_features = data[config['exp_params']['external_features']].to_numpy()
    testing_data = data.iloc[testing_index]
    testing_predictions = -1 * np.ones([len(testing_data)])

    for iter, index in enumerate(testing_index):
        if data[f'y{task_name[0]}_t+{horizon_val}(flag)'][index] == 0:
            continue
        # print(f'{iter}/{len(testing_index)}')
        history_y_t = total_y_t[index-sliding_window:index]
        lag_y = total_y_t[index+1-p:index+1]
        if param_dict['external_feature_flag']:
            history_features = total_features[index+1-sliding_window:index+1]
            future_features = total_features[index+1:index+1+horizon_val]
            model = VAR(endog=history_y_t, exog=history_features)
            model_fit = model.fit(p)
            prediction = model_fit.forecast(y=lag_y, steps=horizon_val, exog_future=future_features)[-1]
        else:
            model = VAR(endog=history_y_t)
            model_fit = model.fit(p)
            prediction = model_fit.forecast(y=lag_y, steps=horizon_val)[-1]
        testing_predictions[iter] = prediction[list(task_prediction_horizon.keys()).index(task_name)]
    return testing_predictions

def grid_search_arima(config, num_files=3):
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

    """
    getting validation results
    """
    variate_options = list()
    for one_variate in config['exp_params']['variate']:
        if one_variate == 'multi':
            variate_options.append({'variate': 'multi'})
        elif one_variate == 'uni':
            uni_param = {'d': config['exp_params']['d_values'], 'q': config['exp_params']['q_values'], 'variate': ['uni']}
            variate_options += list(ParameterGrid(uni_param))

    param_grid = {
        'sliding_window': config['exp_params']['sliding_window'],
        'p': config['exp_params']['p_values'],
        # 'd': config['exp_params']['d_values'],
        # 'q': config['exp_params']['q_values'],
        # 'variate': config['exp_params']['variate'],
        'variate_options': variate_options,
        'external_feature_flag': config['exp_params']['external_feature_flag'],
    }
    param_dict_list = list(ParameterGrid(param_grid))

    if not config['exp_params']['test_flag']:
        print('training...')
        summary = dict()
        for task_index, task_file in enumerate(file_list):
            print(f'{task_index}/{len(file_list)}')
            cur_summary = arima_validation(os.path.join(data_folder, task_file), param_dict_list, config)
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
        predictions = arima_testing(os.path.join(data_folder, task_file), param_summary[task_file], config, interval_multiplier)
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
    parser.add_argument('--gpus', '-g', type=str) #, default='[1]')

    parser.add_argument('--sliding_window', '-sliding_window', type=str, help='list of sliding_window for arima')
    parser.add_argument('--selection_metric', '-selection_metric', type=str, help='metrics to select hyperparameters, one of [RMSE, MAE, MAPE]',)
    parser.add_argument('--external_feature_flag', '-external_feature_flag', type=bool, help='whether to consider external features')
    parser.add_argument('--external_features', '-external_features', type=str, help='list of external feature name list')

    # model-specific features
    parser.add_argument('--p_values', '-p_values', type=str, help='list of p_values for arima')#, default='[1, 2]')
    parser.add_argument('--d_values', '-d_values', type=str, help='list of d_values for arima')#, default='[1]')
    parser.add_argument('--q_values', '-q_values', type=str, help='list of q_values for arima')#, default='[1]')

    parser.add_argument('--variate', '-variate', type=str, help='list of variate options, either uni or multi')
    args = vars(parser.parse_args())
    # print(f'args: {args}')
    with open('./../configs/arima.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    # search for hyperparameters
    grid_search_arima(config, num_files=args['num_files'])
