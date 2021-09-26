# Created by xunannancy at 2021/9/25
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from collections import OrderedDict
import os
import pandas as pd
from tqdm import tqdm
import json
import yaml
import joblib
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools
from torch import optim
from copy import deepcopy

task_prediction_horizon = OrderedDict({
    'load': [60, 1440],
    'wind': [5, 30],
    'solar': [5, 30],
})
task_horizon_list = list()
for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
    for horizon_val in task_prediction_horizon_list:
        task_horizon_list.append(f'y{task_name[0]}_t+{horizon_val}')

prediction_interval_multiplier = {
    '0.99': 2.58,
    '0.98': 2.33,
    '0.97': 2.17,
    '0.96': 2.05,
    '0.95': 1.96,
    '0.90': 1.64,
    '0.85': 1.44,
    '0.80': 1.28,
    '0.75': 1.15,
    '0.70': 1.04,
    '0.65': 0.93,
    '0.60': 0.84,
    '0.55': 0.76,
    '0.50': 0.67
}

def run_evaluate_V3(config, verbose=True):
    if verbose:
        saved_folder = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
        flag = True
        while flag:
            if not os.path.exists(saved_folder):
                os.makedirs(saved_folder)
                last_version = -1
            else:
                last_version = sorted([int(i.split('_')[1]) for i in os.listdir(saved_folder) if i.startswith('version_')])
                if len(last_version) == 0:
                    last_version = -1
                else:
                    last_version = last_version[-1]
            log_dir = os.path.join(saved_folder, f'version_{last_version+1}')
            try:
                os.makedirs(log_dir)
                flag = False
            except:
                flag = True
        print(f'log_dir: {log_dir}')

    data_folder = '/meladyfs/newyork/nanx/freetime/freetime/PowerSystem/processed_datasets/forecasting'

    gt_file_dict = dict()
    for i in os.listdir(data_folder):
        cur_year = int(i.split('.')[0].split('_')[-1])
        if cur_year not in gt_file_dict:
            gt_file_dict[cur_year] = [i]
        else:
            gt_file_dict[cur_year].append(i)

    summary = dict()
    for year, file_list in gt_file_dict.items():
        print(f'year: {year}')
        summary[year] = dict()
        file_counter = 0
        total_results = dict()
        for gt_file in tqdm(file_list):
            if not os.path.exists(os.path.join(config['exp_params']['prediction_path'], gt_file)):
                continue
            cur_results = perform_evaluate_V3(os.path.join(data_folder, gt_file), os.path.join(config['exp_params']['prediction_path'], gt_file))
            for key, val in cur_results.items():
                gt, pred_mean, pred_U, pred_L = val[0], val[1], val[2], val[3]
                if key not in total_results:
                    total_results[key] = [gt, pred_mean, pred_U, pred_L]
                else:
                    total_results[key][0] = np.concatenate([total_results[key][0], gt])
                    total_results[key][1] = np.concatenate([total_results[key][1], pred_mean])
                    total_results[key][2] = np.concatenate([total_results[key][2], pred_U])
                    total_results[key][3] = np.concatenate([total_results[key][3], pred_L])

            file_counter += 1
        for key, val in total_results.items():
            gt, pred_mean, pred_U, pred_L = val[0], val[1], val[2], val[3]
            RMSE = np.sqrt(mean_squared_error(gt, pred_mean))
            MAE = mean_absolute_error(gt, pred_mean)
            MAPE = mean_absolute_percentage_error(gt, pred_mean)
            a = config['exp_params']['prediction_interval']
            term1 = pred_U - pred_L
            term2 = 2./a * (pred_L - gt) * (gt < pred_L)
            term3 = 2./a * (gt - pred_U) * (gt > pred_U)
            MSIS = np.mean(term1 + term2 + term3)# / config['exp_params']['naive_scale'][int(horizon_val.split('+')[1])]
            summary[year][key] = {
                'locs': file_counter,
                'RMSE': RMSE,
                'MAE': MAE,
                'MAPE': MAPE,
                'MSIS': MSIS
            }
    summary = OrderedDict(sorted(summary.items()))
    print(f'summary: {summary}')
    if verbose:
        with open(os.path.join(log_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    return


def perform_evaluate_V3(gt_file, pred_file):
    gt_data = pd.read_csv(gt_file)
    train_flag = gt_data['train_flag'].to_numpy()
    testing_index = sorted(np.argwhere(train_flag == 0).reshape([-1]))
    gt_testing_data = gt_data.iloc[testing_index]

    pred_data = pd.read_csv(pred_file)

    # combine
    merged_results = pd.merge(left=gt_testing_data, right=pred_data, how='left', on='ID')

    results = dict()

    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        for horizon_index, horizon_val in enumerate(task_prediction_horizon_list):
            cur_gt_val, cur_gt_flag, cur_pred_mean = merged_results[f'y{task_name[0]}_t+{horizon_val}(val)'].to_numpy(), merged_results[f'y{task_name[0]}_t+{horizon_val}(flag)'].to_numpy(), merged_results[f'y{task_name[0]}_t+{horizon_val}(mean)'].to_numpy()
            cur_pred_U, cur_pred_L = merged_results[f'y{task_name[0]}_t+{horizon_val}(U)'].to_numpy(), merged_results[f'y{task_name[0]}_t+{horizon_val}(L)'].to_numpy()
            selected_index = sorted(np.argwhere(cur_gt_flag == 1).reshape([-1]))
            valid_gt = cur_gt_val[selected_index]
            val_pred_mean = cur_pred_mean[selected_index]
            val_pred_U, val_pred_L = cur_pred_U[selected_index], cur_pred_L[selected_index]
            results[f'y{task_name[0]}_t+{horizon_val}'] = [valid_gt, val_pred_mean, val_pred_U, val_pred_L]
    return results

def run_evaluate(results_dict, selection_metric):
    assert selection_metric in ['RMSE', 'MAE', 'MAPE']
    summary = dict()
    for task_horizon, res in results_dict.items():
        summary[task_horizon] = list()
        gt, pred = res[0], res[1]
        if selection_metric == 'RMSE':
            val = np.sqrt(mean_squared_error(gt, pred))
        elif selection_metric == 'MAE':
            val = mean_absolute_error(gt, pred)
        elif selection_metric == 'MAPE':
            val = mean_absolute_percentage_error(gt, pred)
        residual = np.array(gt) - np.array(pred)
        std = np.std(residual)
        if len(res) == 2:
            summary[task_horizon] = [val, std]
        elif len(res) == 3:
            summary[task_horizon] = [val, std, res[2]]
    return summary

def merge_parameters(args, config):
    for key, val in config.items():
        if val is None:
            continue
        for subkey, subval in val.items():
            if subkey in args and args[subkey] is not None:
                try:
                    config[key][subkey] = eval(args[subkey])
                except:
                    config[key][subkey] = args[subkey]
    return config


def sklearn_validation(file, param_dict_list, config, log_dir):
    data = pd.read_csv(file)
    train_flag = data['train_flag'].to_numpy()
    training_validation_index = sorted(np.argwhere(train_flag == 1).reshape([-1]))

    summary = dict()
    for param_index, param_dict in tqdm(enumerate(param_dict_list), total=len(param_dict_list)):
        if param_dict['variate'] == 'uni':
            cur_summary = uni_sklearn_validation(data, training_validation_index, param_dict, config, log_dir)
        elif param_dict['variate'] == 'multi':
            if config['model_params']['model_name'] in ['svr', 'gradient_boosting']:
                cur_summary = multi_sklearn_validation_SO(data, training_validation_index, param_dict, config, log_dir)
            elif config['model_params']['model_name'] in ['random_forest', 'linear_regression']:
                cur_summary = multi_sklearn_validation_MO(data, training_validation_index, param_dict, config, log_dir)

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
            'std': param_res['results'][selected_index][1],
            'setting_name': param_res['results'][selected_index][2],
        }
    return selected_summary


def uni_sklearn_validation(data, training_validation_index, param_dict, config, log_dir):
    results = dict()
    sliding_window = param_dict['sliding_window']
    external_feature_flag = param_dict['external_feature_flag']
    if external_feature_flag:
        training_validation_features = data[config['exp_params']['external_features']].to_numpy()[training_validation_index]

    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        training_validation_y_t = data[f'y{task_name[0]}_t'].to_numpy()[training_validation_index]
        history_y_t = list()
        for index in range(sliding_window):
            history_y_t.append(np.expand_dims(training_validation_y_t[index:-sliding_window+index], axis=-1))
            if external_feature_flag:
                history_y_t.append(training_validation_features[index:-sliding_window+index])
        history_y_t.append(np.expand_dims(training_validation_y_t[sliding_window:], axis=-1))
        if external_feature_flag:
            history_y_t.append(training_validation_features[sliding_window:])
        history_y_t = np.concatenate(history_y_t, axis=-1)
        for horizon in task_prediction_horizon_list:
            print(f'uni: {task_name}-{horizon}')
            results[f'y{task_name[0]}_t+{horizon}'] = [[], []]
            training_validation_target_val, training_validation_target_flag = data[f'y{task_name[0]}_t+{horizon}(val)'].to_numpy()[training_validation_index[sliding_window:]], data[f'y{task_name[0]}_t+{horizon}(flag)'].to_numpy()[training_validation_index[sliding_window:]]
            selected_index = np.argwhere(training_validation_target_flag == 1).reshape([-1])
            # if task_name == 'wind':
            #     num_train = int(len(selected_index) * config['exp_params']['train_valid_ratio'] * 1.01)
            # elif task_name == 'solar':
            #     num_train = int(len(selected_index) * (config['exp_params']['train_valid_ratio'] + 0.005))
            # else:
            #     num_train = int(len(selected_index) * config['exp_params']['train_valid_ratio'])
            num_train = config['exp_params']['train_valid_ratio']
            if config['model_params']['model_name'] == 'svr':
                kernel = param_dict['kernel']
                model = SVR(kernel=kernel)
                # model = LinearSVR()
                saved_model_name = f'model_T{task_name[0]}_H{horizon}_S{sliding_window}_K{kernel}_E{external_feature_flag}_U.joblib'
            elif config['model_params']['model_name'] == 'random_forest':
                n_estimators = param_dict['n_estimators']
                criterion = param_dict['criterion']
                model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, random_state=config['logging_params']['manual_seed'])
                saved_model_name = f'model_T{task_name[0]}_H{horizon}_S{sliding_window}_N{n_estimators}_C{criterion}_E{external_feature_flag}_U.joblib'
            elif config['model_params']['model_name'] == 'gradient_boosting':
                learning_rate = param_dict['learning_rate']
                n_estimators = param_dict['n_estimators']
                model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators, random_state=config['logging_params']['manual_seed'])
                saved_model_name = f'model_T{task_name[0]}_H{horizon}_S{sliding_window}_L{learning_rate}_N{n_estimators}_E{external_feature_flag}_U.joblib'
            elif config['model_params']['model_name'] == 'linear_regression':
                normalize = param_dict['normalize']
                model = LinearRegression(normalize=normalize)
                saved_model_name = f'model_T{task_name[0]}_H{horizon}_S{sliding_window}_N{normalize}_E{external_feature_flag}_U.joblib'
            model.fit(history_y_t[selected_index[-num_train*2:-num_train]], training_validation_target_val[selected_index[-num_train*2:-num_train]])
            predictions = model.predict(history_y_t[selected_index[-num_train:]])
            results[f'y{task_name[0]}_t+{horizon}'] = [
                training_validation_target_val[selected_index[-num_train:]],
                predictions,
                saved_model_name
            ]
            joblib.dump(model, os.path.join(log_dir, saved_model_name))
    summary = run_evaluate(results, config['exp_params']['selection_metric'])
    return summary

def multi_sklearn_validation_SO(data, training_validation_index, param_dict, config, log_dir):
    results = dict()
    external_feature_flag = param_dict['external_feature_flag']
    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        for horizon in task_prediction_horizon_list:
            results[f'y{task_name[0]}_t+{horizon}'] = [[], []]
    training_validation_y_t = data[[f'y{task_name[0]}_t' for task_name in task_prediction_horizon.keys()]].to_numpy()[training_validation_index]
    if external_feature_flag:
        training_validation_features = data[config['exp_params']['external_features']].to_numpy()[training_validation_index]

    sliding_window = param_dict['sliding_window']
    history_y_t = list()
    for index in range(sliding_window):
        history_y_t.append(training_validation_y_t[index:-sliding_window+index])
        if external_feature_flag:
            history_y_t.append(training_validation_features[index:-sliding_window+index])
    history_y_t.append(training_validation_y_t[sliding_window:])
    if external_feature_flag:
        history_y_t.append(training_validation_features[sliding_window:])
    history_y_t = np.concatenate(history_y_t, axis=-1)

    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        for horizon_index, horizon in enumerate(task_prediction_horizon_list):
            print(f'multi-SO: {task_name}-{horizon}')
            training_validation_target_val = data[f'y{task_name[0]}_t+{horizon}(val)'].to_numpy()[training_validation_index[sliding_window:]]
            training_validation_target_flag = data[f'y{task_name[0]}_t+{horizon}(flag)'].to_numpy()[training_validation_index[sliding_window:]]
            selected_index = np.argwhere(training_validation_target_flag == 1).reshape([-1])
            # if task_name == 'wind':
            #     num_train = int(len(selected_index) * config['exp_params']['train_valid_ratio'] * 1.01)
            # elif task_name == 'solar':
            #     num_train = int(len(selected_index) * (config['exp_params']['train_valid_ratio'] + 0.005))
            # else:
            #     num_train = int(len(selected_index) * config['exp_params']['train_valid_ratio'])
            num_train = config['exp_params']['train_valid_ratio']

            if config['model_params']['model_name'] == 'svr':
                kernel = param_dict['kernel']
                model = SVR(kernel=kernel)
                # model = LinearSVR()
                saved_model_name = f'model_T{task_name[0]}_H{horizon}_S{sliding_window}_K{kernel}_E{external_feature_flag}_MO.joblib'
            elif config['model_params']['model_name'] == 'gradient_boosting':
                learning_rate = param_dict['learning_rate']
                n_estimators = param_dict['n_estimators']
                model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators)
                saved_model_name = f'model_T{task_name[0]}_H{horizon}_S{sliding_window}_L{learning_rate}_N{n_estimators}_E{external_feature_flag}_MO.joblib'
            model.fit(history_y_t[selected_index[-num_train*2:-num_train]], training_validation_target_val[selected_index[-num_train*2:-num_train]])
            predictions = model.predict(history_y_t[selected_index[-num_train:]])
            results[f'y{task_name[0]}_t+{horizon}'] = [
                training_validation_target_val[selected_index[-num_train:]],
                predictions,
                saved_model_name
                ]
            joblib.dump(model, os.path.join(log_dir, saved_model_name))
    summary = run_evaluate(results, config['exp_params']['selection_metric'])
    return summary

def multi_sklearn_validation_MO(data, training_validation_index, param_dict, config, log_dir):
    print('multi-mo:')
    external_feature_flag = param_dict['external_feature_flag']
    results = dict()
    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        for horizon in task_prediction_horizon_list:
            results[f'y{task_name[0]}_t+{horizon}'] = [[], []]
    training_validation_y_t = data[[f'y{task_name[0]}_t' for task_name in task_prediction_horizon.keys()]].to_numpy()[training_validation_index]
    if external_feature_flag:
        training_validation_features = data[config['exp_params']['external_features']].to_numpy()[training_validation_index]

    sliding_window = param_dict['sliding_window']
    history_y_t = list()
    for index in range(sliding_window):
        history_y_t.append(training_validation_y_t[index:-sliding_window+index])
        if external_feature_flag:
            history_y_t.append(training_validation_features[index:-sliding_window+index])
    history_y_t.append(training_validation_y_t[sliding_window:])
    if external_feature_flag:
        history_y_t.append(training_validation_features[sliding_window:])
    history_y_t = np.concatenate(history_y_t, axis=-1)

    flag_prod = 0
    target_columns = list()
    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        for horizon in task_prediction_horizon_list:
            cur_flag = data[f'y{task_name[0]}_t+{horizon}(flag)'].to_numpy()[training_validation_index[sliding_window:]]
            flag_prod += cur_flag
            target_columns.append(f'y{task_name[0]}_t+{horizon}(val)')
    training_validation_target_val = data[target_columns].to_numpy()[training_validation_index[sliding_window:]]
    selected_index = np.argwhere(flag_prod > 0).reshape([-1])
    # num_train = int(len(selected_index) * config['exp_params']['train_valid_ratio'])
    num_train = config['exp_params']['train_valid_ratio']
    if config['model_params']['model_name'] == 'random_forest':
        n_estimators = param_dict['n_estimators']
        criterion = param_dict['criterion']
        model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, random_state=config['logging_params']['manual_seed'])
        saved_model_name = f'model_S{sliding_window}_N{n_estimators}_C{criterion}_E{external_feature_flag}_MO.joblib'
    elif config['model_params']['model_name'] == 'linear_regression':
        normalize = param_dict['normalize']
        model = LinearRegression(normalize=normalize)
        saved_model_name = f'model_S{sliding_window}_N{normalize}_E{external_feature_flag}_MO.joblib'
    model.fit(history_y_t[selected_index[-num_train*2:-num_train]], training_validation_target_val[selected_index[-num_train*2:-num_train]])
    predictions = model.predict(history_y_t[selected_index[-num_train:]])
    counter = 0
    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        for horizon in task_prediction_horizon_list:
            results[f'y{task_name[0]}_t+{horizon}'] = [
                training_validation_target_val[selected_index[-num_train:]][:, counter],
                predictions[:, counter],
                saved_model_name
                ]
            counter += 1
        joblib.dump(model, os.path.join(log_dir, saved_model_name))
    summary = run_evaluate(results, config['exp_params']['selection_metric'])
    return summary

def sklearn_testing(file, param_summary, config, interval_multiplier, log_dir):
    data = pd.read_csv(file)
    train_flag = data['train_flag'].to_numpy()
    testing_index = np.array(sorted(np.argwhere(train_flag == 0).reshape([-1])))
    testing_data = data.iloc[testing_index]
    testing_ID = testing_data['ID'].to_numpy()

    testing_predictions, columns = list(), list()
    for idx, (task_horizon, res) in enumerate(param_summary.items()):
        print(f'task_horizon: {task_horizon}')
        task_name = task_horizon.split('_')[0]
        task_name = {
            'yl': 'load',
            'yw': 'wind',
            'ys': 'solar'
        }[task_name]
        param_dict, std, saved_model_name = res['param_dict'], res['std'], res['setting_name']
        if param_dict['variate'] == 'uni':
            predictions = uni_sklearn_testing(task_name, data, testing_index, param_dict, config, log_dir, saved_model_name)
        elif param_dict['variate'] == 'multi':
            predictions = multi_sklearn_testing(data, testing_index, param_dict, config, log_dir, saved_model_name)
            if config['model_params']['model_name'] in ['random_forest', 'linear_regression']:
                predictions = predictions[:, task_horizon_list.index(task_horizon)]
        testing_predictions.append(predictions)
        columns.append(f'{task_horizon}(mean)')

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

def uni_sklearn_testing(task_name, data, testing_index, param_dict, config, log_dir, saved_model_name):
    if param_dict['external_feature_flag']:
        total_features = data[config['exp_params']['external_features']].to_numpy()

    total_y_t = data[f'y{task_name[0]}_t'].to_numpy()

    sliding_window = param_dict['sliding_window']
    model = joblib.load(os.path.join(log_dir, saved_model_name))

    history_y_t = list()
    for index in range(sliding_window, 0, -1):
        history_y_t.append(np.expand_dims(total_y_t[testing_index-index], axis=-1))
        if param_dict['external_feature_flag']:
            history_y_t.append(total_features[testing_index-index])
    history_y_t.append(np.expand_dims(total_y_t[testing_index], axis=-1))
    if param_dict['external_feature_flag']:
        history_y_t.append(total_features[testing_index])
    history_y_t = np.concatenate(history_y_t, axis=-1)
    predictions = model.predict(history_y_t)
    return predictions

def multi_sklearn_testing(data, testing_index, param_dict, config, log_dir, saved_model_name):
    if param_dict['external_feature_flag']:
        total_features = data[config['exp_params']['external_features']].to_numpy()

    total_y_t = data[[f'y{l[0]}_t' for l in task_prediction_horizon.keys()]].to_numpy()
    sliding_window = param_dict['sliding_window']
    model = joblib.load(os.path.join(log_dir, saved_model_name))

    history_y_t = list()
    for index in range(sliding_window, 0, -1):
        history_y_t.append(total_y_t[testing_index-index])
        if param_dict['external_feature_flag']:
            history_y_t.append(total_features[testing_index-index])
    history_y_t.append(total_y_t[testing_index])
    if param_dict['external_feature_flag']:
        history_y_t.append(total_features[testing_index])
    history_y_t = np.concatenate(history_y_t, axis=-1)
    predictions = model.predict(history_y_t)
    return predictions

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %.3f M' % (num_params/1000000))
    return

class Pytorch_DNN_exp(pl.LightningModule):
    def __init__(self, file, param_dict, config):
        super().__init__()
        self.save_hyperparameters()

        self.param_dict = param_dict
        self.config = config
        self.max_epochs = config['trainer_params']['max_epochs']

    def oracle_loss(self, batch):
        loss, pred = self.model.loss_function(batch)
        if self.eval():
            # validation
            _, y, flag = batch
            pred, y = pred.detach().cpu().numpy(), y.detach().cpu().numpy()
            if self.param_dict['normalization'] != 'none':
                pred = self.dataloader.scalar_y.inverse_transform(pred)
                y = self.dataloader.scalar_y.inverse_transform(y)
            # selected_index = torch.where(flag.reshape([-1]) == 1)[0].detach().cpu().numpy()
            pred_list, y_list = list(), list()
            for column_index in range(len(self.dataloader.target_val_column_names)):
                selected_index = torch.where(flag[:, column_index] == 1)[0].detach().cpu().numpy()
                pred_list.append(pred[selected_index, column_index])
                y_list.append(y[selected_index, column_index])
            return {'loss': loss, 'pred': pred_list, 'y': y_list}
            # return {'loss': loss, 'pred': pred.reshape([-1])[selected_index], 'y': y.reshape([-1])[selected_index]}
        else:
            # training
            return {'loss': loss}

    def training_step(self, batch, batch_idx):
        train_loss = self.oracle_loss(batch)
        self.logger.experiment.log({f'train_{key}': float(val) for key, val in train_loss.items() if key not in ['pred', 'y']})
        return train_loss

    def training_epoch_end(self, outputs):
        avg_metric_values = {
            'loss': list(),
        }

        for output in outputs:
            for key in avg_metric_values.keys():
                avg_metric_values[key].append(output[key])
        for metric, avg_value in avg_metric_values.items():
            self.log(f'avg_train_{metric}', torch.mean(torch.stack(avg_value)))
        return

    def validation_step(self, batch, batch_idx):
        valid_loss = self.oracle_loss(batch)
        self.logger.experiment.log({f'val_{key}': float(val) for key, val in valid_loss.items() if key not in ['pred', 'y']})
        return valid_loss

    def validation_epoch_end(self, outputs):
        avg_metric_values = {
            'loss': list(),
        }
        pred = [list() for i in range(len(self.dataloader.target_val_column_names))]
        gt = deepcopy(pred)

        for output in outputs:
            for key in avg_metric_values.keys():
                avg_metric_values[key].append(output[key])
            # pred += list(output['pred'].cpu().detach().numpy())
            # gt += list(output['y'].cpu().detach().numpy())
            for i in range(len(self.dataloader.target_val_column_names)):
                pred[i] += list(output['pred'][i])
                gt[i] += list(output['y'][i])

        for metric, avg_value in avg_metric_values.items():
            self.log(f'avg_val_{metric}', torch.mean(torch.stack(avg_value)))

        pred_flatten, gt_flatten = np.array(list(itertools.chain(*pred))), np.array(list(itertools.chain(*gt)))
        if self.config['exp_params']['selection_metric'] == 'RMSE':
            val = np.sqrt(mean_squared_error(gt_flatten, pred_flatten))
        elif self.config['exp_params']['selection_metric'] == 'MAE':
            val = mean_absolute_error(gt_flatten, pred_flatten)
        elif self.config['exp_params']['selection_metric'] == 'MAPE':
            val = mean_absolute_percentage_error(gt, pred_flatten)
        self.log('avg_val_metric', val)
        # # std
        # if self.config['model_params']['model_name'] not in ['DeepAR']:
        #     residual = np.array(gt) - np.array(pred)
        #     std = np.std(residual)
        #     self.log('std', std)

        cur_logger_folder = f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
        if self.current_epoch == self.max_epochs - 1:
            self.plot_write_summmary_train_valid(cur_logger_folder)
        # save standard deviation
        if self.config['model_params']['model_name'] not in ['DeepAR']:
            std = [np.std(np.array(i) - np.array(j)) for i, j in zip(gt, pred)]
            with open(os.path.join(cur_logger_folder, 'std.txt'), 'a+') as f:
                f.writelines(str(self.current_epoch)+' '+'_'.join(map(str, std))+'\n')
        return

    def test_step(self, batch, batch_idx):
        ID, x = batch
        prediction = self.model.forward(x)
        if self.param_dict['normalization'] == 'none' and self.config['model_params']['model_name'] not in ['cnn']:
            prediction = torch.exp(prediction)
        return ID, prediction

    def test_epoch_end(self, outputs):
        interval_multiplier = prediction_interval_multiplier[str(self.config['exp_params']['prediction_interval'])]
        IDs, predictions = list(), list()
        for output in outputs:
            IDs += list(output[0].cpu().detach().numpy())
            predictions.append(output[1].cpu().detach().numpy())
        IDs, predictions = np.array(IDs), np.concatenate(predictions, axis=0)
        if self.param_dict['normalization'] != 'none':
            predictions = self.dataloader.scalar_y.inverse_transform(predictions)
        cur_logger_folder = f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
        std = self.param_dict['std']

        predictions_frame = pd.DataFrame(
            data=predictions,
            columns=[i.replace('val', 'mean') for i in self.dataloader.target_val_column_names],
        )
        predictions_frame['ID'] = IDs
        for column_index, one_target_column in enumerate(self.dataloader.target_val_column_names):
            suffix = one_target_column.split('(')[0]
            predictions_frame[f'{suffix}(U)'] = predictions_frame[f'{suffix}(mean)'] + interval_multiplier * std[column_index]
            predictions_frame[f'{suffix}(L)'] = predictions_frame[f'{suffix}(mean)'] - interval_multiplier * std[column_index]

        predictions_frame.to_csv(os.path.join(cur_logger_folder, self.dataloader.file.split('/')[-1]), index=False)

        return

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.param_dict['learning_rate']
        )
        optims.append(optimizer)
        return optims, scheds

    def plot_write_summmary_train_valid(self, log_dir):
        df = pd.read_csv(os.path.join(log_dir, 'metrics.csv'))
        key_loss_name = {
            'loss': 'r',
        }

        fig, axes = plt.subplots(2, 2, figsize=(4*2, 3*2))
        for value_index, value_type in enumerate(['step', 'epoch']):
            if value_type == 'step':
                prefix = ''
            elif value_type == 'epoch':
                prefix = 'avg_'
            ax = axes[value_index]
            for data_index, data_type in enumerate(['train', 'val']):
                if value_type == 'step' and data_type == 'val':
                    sanity_check = 5
                else:
                    sanity_check = 0
                ax = axes[value_index, data_index]

                for key, color in key_loss_name.items():
                    cur_value = df[f'{prefix}{data_type}_{key}'].dropna().to_numpy()[sanity_check:]
                    ax.plot(range(1, len(cur_value)+1), cur_value, color=color, label=key)

                ax.set_xlabel(f'{value_type.capitalize()}s')
                ax.set_ylabel('Loss')
                ax.set_title(f'{data_type}: {len(cur_value)} {value_type}s')
                # ax.legend(loc='upper right')
        handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles,  # The line objects
        #            labels,  # The labels for each line
        #            loc="lower center",  # Position of legend
        #            borderaxespad=0.1, ncol=3)  # Small spacing around legend box)
        fig.legend(handles, labels, bbox_to_anchor=(0, -0.1, 1, 1), bbox_transform=plt.gcf().transFigure)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'loss.png'))
        # plt.show()
        plt.close()
        return

    def train_dataloader(self):
        return self.dataloader.load_train()

    def val_dataloader(self):
        return self.dataloader.load_valid()

    def test_dataloader(self):
        return self.dataloader.load_test()

def Pytorch_DNN_validation(file, param_dict_list, log_dir, config, model_exp):
    for param_index, param_dict in enumerate(param_dict_list):
        param_dict = OrderedDict(param_dict)
        setting_name = 'param'
        for key, val in param_dict.items():
            setting_name += f'_{key[0].capitalize()}{val}'

        tt_logger = TestTubeLogger(
            save_dir=log_dir,
            name=setting_name,
            debug=False,
            create_git_tag=False,
            version=0
        )
        # if config['model_params']['model_name'] not in ['DeepAR']:
        #     checkpoint_callback_V = ModelCheckpoint(
        #         dirpath=f"{tt_logger.save_dir}/{tt_logger.name}/version_0/",
        #         filename='best-{epoch:02d}-{avg_val_metric:.3f}-{std:.3f}',
        #         save_top_k=1,
        #         verbose=True,
        #         mode='min',
        #         monitor='avg_val_metric',
        #     )
        # else:
        checkpoint_callback_V = ModelCheckpoint(
            dirpath=f"{tt_logger.save_dir}/{tt_logger.name}/version_0/",
            filename='best-{epoch:02d}-{avg_val_metric:.3f}',
            save_top_k=1,
            verbose=True,
            mode='min',
            monitor='avg_val_metric',
        )
        exp = model_exp(
            file=file,
            param_dict=param_dict,
            config=config
        )
        runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                         min_epochs=1,
                         logger=tt_logger,
                         log_every_n_steps=50,
                         limit_train_batches=1.,
                         limit_val_batches=1.,
                         limit_test_batches=0,
                         num_sanity_val_steps=5,
                         checkpoint_callback=True,
                         callbacks=[checkpoint_callback_V],
                         max_epochs=config['trainer_params']['max_epochs'],
                         gpus=config['trainer_params']['gpus']
                         )
        runner.fit(exp)
    return

def Pytorch_DNN_testing(file, param_dict, log_dir, config, model_exp):
    param_dict = OrderedDict(param_dict)
    setting_name = 'param'
    for key, val in param_dict.items():
        if key == 'std':
            continue
        setting_name += f'_{key[0].capitalize()}{val}'

    tmp = log_dir[log_dir.find(config['logging_params']['name'])+len(config['logging_params']['name'])+1:]
    tt_logger = TestTubeLogger(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False,
        version=int(tmp.split('/')[0].split('_')[1])
    )
    runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                     min_epochs=1,
                     logger=tt_logger,
                     log_every_n_steps=50,
                     limit_train_batches=0,
                     limit_val_batches=0,
                     limit_test_batches=1.,
                     num_sanity_val_steps=0,
                     checkpoint_callback=False,
                     max_epochs=1,
                     gpus=config['trainer_params']['gpus']
                     )
    # model_path = [i for i in os.listdir(f"{tt_logger.save_dir}/{tt_logger.name}/version_{tt_logger.version}/{setting_name}/version_0") if i.endswith('.ckpt') and 'best' in i]
    model_path = [i for i in os.listdir(f"{log_dir}/{setting_name}/version_0") if i.endswith('.ckpt') and 'best' in i]
    exp = model_exp(
        file=file,
        param_dict=param_dict,
        config=config
    )
    checkpoint_path = os.path.join(f"{log_dir}/{setting_name}/version_0", model_path[0])
    exp = exp.load_from_checkpoint(checkpoint_path)
    print(f'Load checkpoint from {checkpoint_path}...')
    exp.param_dict = param_dict
    runner.test(exp)

    return
