# Created by xunannancy at 2021/9/25
"""
naive: keep y_t for y_t+h
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import pandas as pd
import yaml
import numpy as np
from tqdm import tqdm
import itertools
from utils import merge_parameters, prediction_interval_multiplier, task_prediction_horizon, run_evaluate_V3

def single_naive(file, prediction_horizon_list, interval_multiplier):
    data = pd.read_csv(file)
    train_flag = data['train_flag'].to_numpy()
    training_index = sorted(np.argwhere(train_flag == 1).reshape([-1]))
    testing_index = sorted(np.argwhere(train_flag == 0).reshape([-1]))

    testing_data = data.iloc[testing_index]
    testing_y_t = testing_data['y_t'].to_numpy()
    testing_ID = testing_data['ID'].to_numpy()

    predictions = pd.DataFrame(
        data=np.repeat(testing_y_t.reshape([-1, 1]), len(prediction_horizon_list), axis=-1),
        columns=[f'y_t+{i}(mean)' for i in prediction_horizon_list],
    )

    """
    prediction interval
    """
    training_data = data.iloc[training_index]
    prediction = training_data['y_t'].to_numpy()
    for one_horizon in prediction_horizon_list:
        gt = training_data[f'y_t+{one_horizon}(val)'].to_numpy()
        flag = training_data[f'y_t+{one_horizon}(flag)'].to_numpy()
        residual = gt - prediction
        std = np.std(residual[np.argwhere(flag == 1).reshape([-1])])
        predictions[f'y_t+{one_horizon}(U)'] = predictions[f'y_t+{one_horizon}(mean)'] + interval_multiplier * std
        predictions[f'y_t+{one_horizon}(L)'] = predictions[f'y_t+{one_horizon}(mean)'] - interval_multiplier * std

    predictions['ID'] = testing_ID

    return predictions

def multiple_naive(file, prediction_horizon_list, interval_multiplier):
    data = pd.read_csv(file)
    train_flag = data['train_flag'].to_numpy()
    training_index = sorted(np.argwhere(train_flag == 1).reshape([-1]))
    testing_index = sorted(np.argwhere(train_flag == 0).reshape([-1]))

    testing_data = data.iloc[testing_index]
    loc_index = sorted([int(i[i.find('(')+1:i.find(')')]) for i in list(testing_data) if i.startswith('y_t(')])
    testing_y_t = testing_data[[f'y_t({i})' for i in loc_index]].to_numpy()
    testing_ID = testing_data['ID'].to_numpy()

    predictions = pd.DataFrame(
        data=np.repeat(np.expand_dims(testing_y_t, axis=1), len(prediction_horizon_list), axis=1).reshape([len(testing_index), -1]),
        columns=list(itertools.chain(*[[f'y_t+{j}({i})(mean)' for i in loc_index] for j in prediction_horizon_list])),
        index=data.index[testing_index].to_numpy()
    )

    training_data = data.iloc[training_index]

    for one_horizon in prediction_horizon_list:
        for one_loc in loc_index:
            prediction = training_data[f'y_t({one_loc})'].to_numpy()
            gt = training_data[f'y_t+{one_horizon}({one_loc})(val)'].to_numpy()
            flag = training_data[f'y_t+{one_horizon}({one_loc})(flag)'].to_numpy()
            residual = gt - prediction
            std = np.std(residual[np.argwhere(flag == 1).reshape([-1])])
            predictions[f'y_t+{one_horizon}({one_loc})(U)'] = predictions[f'y_t+{one_horizon}({one_loc})(mean)'] + interval_multiplier * std
            predictions[f'y_t+{one_horizon}({one_loc})(L)'] = predictions[f'y_t+{one_horizon}({one_loc})(mean)'] - interval_multiplier * std

    predictions['ID'] = testing_ID

    return predictions, loc_index

def run_naive(config):
    interval_multiplier = prediction_interval_multiplier[str(config['exp_params']['prediction_interval'])]
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

    feature_name, location_type = config['exp_params']['feature_name'], config['exp_params']['location_type']
    assert feature_name in ['solar', 'wind', 'load']
    assert location_type in ['single', 'multiple']
    if feature_name in ['solar', 'wind']:
        assert config['exp_params']['prediction_horizon'] == [5, 30]
    elif feature_name == 'load':
        assert config['exp_params']['prediction_horizon'] == [60, 1440]

    # if location_type == 'single':
    #     file_list = sorted([os.path.join(data_folder, i) for i in os.listdir(data_folder) if i.startswith(feature_name) and 'total' not in i])
    #     for loc_index, loc_file in tqdm(enumerate(file_list), total=len(file_list)):
    #         predictions = single_naive(loc_file, config['exp_params']['prediction_horizon'], interval_multiplier)
    #         new_columns = ['ID']
    #         for one_horizon in config['exp_params']['prediction_horizon']:
    #             new_columns += [f'y_t+{one_horizon}(mean)', f'y_t+{one_horizon}(U)', f'y_t+{one_horizon}(L)']
    #         predictions.to_csv(os.path.join(log_dir, f'{feature_name}_{loc_index}.csv'), index=False, columns=new_columns)
    # elif location_type == 'multiple':
    #     file = os.path.join(data_folder, f'{feature_name}_total.csv')
    #     predictions, loc_index = multiple_naive(file, config['exp_params']['prediction_horizon'], interval_multiplier)
    #     new_columns = ['ID']
    #     for one_horizon in config['exp_params']['prediction_horizon']:
    #         for one_loc in loc_index:
    #             new_columns += [f'y_t+{one_horizon}({one_loc})(mean)', f'y_t+{one_horizon}({one_loc})(U)', f'y_t+{one_horizon}({one_loc})(L)']
    #     predictions.to_csv(os.path.join(log_dir, f'{feature_name}_total.csv'), index=False, columns=new_columns)

    file_list = sorted([i for i in os.listdir(data_folder) if i.startswith(feature_name) and 'total' not in i])
    for task_index, task_file in tqdm(enumerate(file_list), total=len(file_list)):
        # predictions = single_naive(loc_file, config['exp_params']['prediction_horizon'], interval_multiplier)
        predictions = perform_naive_V3(os.path.join(data_folder, task_file), interval_multiplier)
        new_columns = ['ID']
        for one_horizon in config['exp_params']['prediction_horizon']:
            new_columns += [f'y_t+{one_horizon}(mean)', f'y_t+{one_horizon}(U)', f'y_t+{one_horizon}(L)']
        predictions.to_csv(os.path.join(log_dir, task_file), index=False, columns=new_columns)

    # save config
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    return

def perform_naive_V3(file, interval_multiplier):
    data = pd.read_csv(file)
    train_flag = data['train_flag'].to_numpy()
    training_index = sorted(np.argwhere(train_flag == 1).reshape([-1]))
    testing_index = sorted(np.argwhere(train_flag == 0).reshape([-1]))

    testing_data = data.iloc[testing_index]
    testing_ID = testing_data['ID'].to_numpy()

    predictions, columns = list(), list()
    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        predictions.append(np.repeat(np.expand_dims(testing_data[f'y{task_name[0]}_t'], axis=1), len(task_prediction_horizon_list), axis=-1))
        columns += [f'y{task_name[0]}_t+{j}(mean)' for j in task_prediction_horizon_list]
    prediction_frame = pd.DataFrame(
        data=np.concatenate(predictions, axis=-1),
        columns=columns,
        index=data.index[testing_index].to_numpy()
    )

    training_data = data.iloc[training_index]
    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        for one_horizon in task_prediction_horizon_list:
            prediction = training_data[f'y{task_name[0]}_t'].to_numpy()
            gt = training_data[f'y{task_name[0]}_t+{one_horizon}(val)'].to_numpy()
            flag = training_data[f'y{task_name[0]}_t+{one_horizon}(flag)'].to_numpy()
            residual = gt - prediction
            std = np.std(residual[np.argwhere(flag == 1).reshape([-1])])
            prediction_frame[f'y{task_name[0]}_t+{one_horizon}(U)'] = prediction_frame[f'y{task_name[0]}_t+{one_horizon}(mean)'] + interval_multiplier * std
            prediction_frame[f'y{task_name[0]}_t+{one_horizon}(L)'] = prediction_frame[f'y{task_name[0]}_t+{one_horizon}(mean)'] - interval_multiplier * std

    prediction_frame['ID'] = testing_ID
    return prediction_frame

def run_naive_V3(config, num_files=3):
    interval_multiplier = prediction_interval_multiplier[str(config['exp_params']['prediction_interval'])]
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
                last_version = sorted([int(i.split('_')[1]) for i in os.listdir(saved_folder) if i.startswith('version_')])
                if len(last_version) == 0:
                    last_version = -1
                else:
                    last_version = last_version[-1]
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
    # file_list = sorted([i for i in os.listdir(data_folder) if 'zone' in i and i.endswith('.csv')])[6:7]
    for task_index, task_file in tqdm(enumerate(file_list), total=len(file_list)):
        if config['exp_params']['test_flag'] and os.path.exists(os.path.join(os.path.join(log_dir, task_file))):
            continue
        predictions = perform_naive_V3(os.path.join(data_folder, task_file), interval_multiplier)
        new_columns = ['ID']
        for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
            for one_horizon in task_prediction_horizon_list:
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
    parser.add_argument('--manual_seed', '-manual_seed', type=int, help='manual_seed')
    parser.add_argument('--num_files', '-num_files', type=int, default=3, help='number of files to predict')
    parser.add_argument('--gpus', '-g', type=str, default='[1]')

    args = vars(parser.parse_args())
    # print(f'args: {args}')
    with open('./../configs/naive.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(f'before merge: config, {config}')
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    run_naive_V3(config, num_files=args['num_files'])

    """
    3-6
    summary: OrderedDict([(2018, {'yl_t+60': {'locs': 1, 'RMSE': 0.07025594272130357, 'MAE': 0.05418589634280538, 'MAPE': 0.05118748441639976, 'MSIS': 0.2831740012883765}, 'yl_t+1440': {'locs': 1, 'RMSE': 0.1401648469298121, 'MAE': 0.11039672527179971, 'MAPE': 0.10499537191590273, 'MSIS': 0.44918565851411874}, 'yw_t+5': {'locs': 1, 'RMSE': 0.0035044568011629697, 'MAE': 0.00118534773612544, 'MAPE': 0.08304780634296398, 'MSIS': 0.01648452773882635}, 'yw_t+30': {'locs': 1, 'RMSE': 0.01076931158344016, 'MAE': 0.0035605365695690868, 'MAPE': 0.21328055238039023, 'MSIS': 0.05031062628127059}, 'ys_t+5': {'locs': 1, 'RMSE': 0.0410320059934712, 'MAE': 0.02291243597950762, 'MAPE': 0.3892178851207519, 'MSIS': 0.11260023461709615}, 'ys_t+30': {'locs': 1, 'RMSE': 0.09529863965699746, 'MAE': 0.07095552363452597, 'MAPE': 0.9779857692942917, 'MSIS': 0.3258431761260312}}), (2019, {'yl_t+60': {'locs': 1, 'RMSE': 0.09685589231877949, 'MAE': 0.0691872839380173, 'MAPE': 0.05928175844086626, 'MSIS': 0.28424281168233634}, 'yl_t+1440': {'locs': 1, 'RMSE': 0.26151426450544646, 'MAE': 0.18754130726958593, 'MAPE': 0.1627000560998403, 'MSIS': 0.6612108842946128}, 'yw_t+5': {'locs': 1, 'RMSE': 0.003636741913448436, 'MAE': 0.0013513814743112392, 'MAPE': 0.08463087801733589, 'MSIS': 0.016210731930486864}, 'yw_t+30': {'locs': 1, 'RMSE': 0.011117666951260782, 'MAE': 0.004172394401537884, 'MAPE': 0.22663887681821063, 'MSIS': 0.051045120853219654}, 'ys_t+5': {'locs': 1, 'RMSE': 0.04062467746492532, 'MAE': 0.0228758752190794, 'MAPE': 0.29241535355317466, 'MSIS': 0.14694674896137316}, 'ys_t+30': {'locs': 1, 'RMSE': 0.09436560102082377, 'MAE': 0.07067941634657159, 'MAPE': 1.4595155032741394, 'MSIS': 0.3694508482041775}}), (2020, {'yl_t+60': {'locs': 1, 'RMSE': 0.06698315766939927, 'MAE': 0.052754393303936324, 'MAPE': 0.04501295098380464, 'MSIS': 0.31581894311390274}, 'yl_t+1440': {'locs': 1, 'RMSE': 0.09854239672598991, 'MAE': 0.06919626242758305, 'MAPE': 0.05907744089833965, 'MSIS': 0.5465983412614699}, 'yw_t+5': {'locs': 1, 'RMSE': 0.003059974596041827, 'MAE': 0.0009596823186293017, 'MAPE': 0.08580439692895196, 'MSIS': 0.01613773312242729}, 'yw_t+30': {'locs': 1, 'RMSE': 0.009090592521440144, 'MAE': 0.0028440133468620923, 'MAPE': 0.2266429807800675, 'MSIS': 0.049737425966051844}, 'ys_t+5': {'locs': 1, 'RMSE': 0.044289903413633314, 'MAE': 0.02291265715381269, 'MAPE': 0.2683804337504674, 'MSIS': 0.1283827277112751}, 'ys_t+30': {'locs': 1, 'RMSE': 0.11050674685917457, 'MAE': 0.08143996239652589, 'MAPE': 1.1429168744596447, 'MSIS': 0.3529323503076099}})])
    6:
    summary: OrderedDict([(2018, {'yl_t+60': {'locs': 1, 'RMSE': 0.045776470690831725, 'MAE': 0.035698054893200944, 'MAPE': 0.03616633568660805, 'MSIS': 0.1797819441042448}, 'yl_t+1440': {'locs': 1, 'RMSE': 0.05743003576847672, 'MAE': 0.03687541713717991, 'MAPE': 0.03775337253885224, 'MSIS': 0.27642641502165516}, 'yw_t+5': {'locs': 1, 'RMSE': 0.004265324602494913, 'MAE': 0.0017022210870919453, 'MAPE': 0.07236734602105888, 'MSIS': 0.0200960638705502}, 'yw_t+30': {'locs': 1, 'RMSE': 0.012883460214231339, 'MAE': 0.004918982968368984, 'MAPE': 0.16879098609080326, 'MSIS': 0.05396616526196156}, 'ys_t+5': {'locs': 1, 'RMSE': 0.04180893693729611, 'MAE': 0.02296606700520501, 'MAPE': 0.2510696013119705, 'MSIS': 0.1406993415685966}, 'ys_t+30': {'locs': 1, 'RMSE': 0.09991880647169718, 'MAE': 0.07257301563605875, 'MAPE': 0.7964635984392058, 'MSIS': 0.37137070312487563}}), (2019, {}), (2020, {})])
    
    """

