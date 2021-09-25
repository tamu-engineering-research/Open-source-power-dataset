# Created by xunannancy at 2021/9/25
"""
DNNs implemented by pytorhc lightning
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import yaml
from utils import merge_parameters, print_network, Pytorch_DNN_exp, Pytorch_DNN_validation, Pytorch_DNN_testing, \
    run_evaluate_V3, task_prediction_horizon
import torch
import torch.nn as nn
import os
from sklearn.model_selection import ParameterGrid
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class HistoryConcatTrainDataset(Dataset):
    def __init__(self, x, y, flag):
        """
        for training & validation dataset
        :param x: historical y and external features
        :param y: future y
        :param flag: whether future y is to predict or not; for loss computation
        """
        self.x = x
        self.y = y
        self.flag = flag

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.flag[idx]

class HistoryConcatTestDataset(Dataset):
    def __init__(self, ID, x):
        """
        for testing
        :param ID: testing ID
        :param x: historical features and y
        """
        self.ID = ID
        self.x = x
    def __len__(self):
        return len(self.ID)
    def __getitem__(self, idx):
        return self.ID[idx], self.x[idx]

class HistoryConcatLoader():
    def __init__(self, file, param_dict, config):
        self.file = file
        self.config = config
        self.param_dict = param_dict

        if self.config['model_params']['model_name'] == 'DeepAR' and self.param_dict['variate'] == 'uni':
            self.task_name = self.config['task_name']
        else:
            self.task_name = None

        self.batch_size = self.param_dict['batch_size']
        self.sliding_window = self.param_dict['sliding_window']

        self.train_valid_ratio = config['exp_params']['train_valid_ratio']
        self.num_workers = config['exp_params']['num_workers']

        if 'variate' not in self.param_dict or self.param_dict['variate'] == 'multi':
            # default DNN handles multiple location
            self.variate = 'multi'
        elif self.param_dict['variate'] == 'uni':
            self.variate = 'uni'

        if 'external_flag' not in self.param_dict or self.param_dict['external_flag'] == True:
            # DNN reads external features by default
            self.external_flag = True
        elif self.param_dict['external_flag'] == False:
            self.external_flag = False
        self.set_dataset()

    def set_dataset(self):
        data = pd.read_csv(self.file)

        """
        prepare training & validation datasets
        """
        train_flag = data['train_flag'].to_numpy()
        training_validation_index = np.sort(np.argwhere(train_flag == 1).reshape([-1]))[self.sliding_window:]

        if self.variate == 'multi':
            self.history_column_names = list()
            self.target_val_column_names = list()
            for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
                self.history_column_names.append(f'y{task_name[0]}_t')
                for horizon_val in task_prediction_horizon_list:
                    self.target_val_column_names.append(f'y{task_name[0]}_t+{horizon_val}(val)')
        elif self.variate == 'uni':
            self.history_column_names = [f'y{self.task_name[0]}_t']
            self.target_val_column_names = [f'y{self.task_name[0]}_t+{horizon}(val)' for horizon in task_prediction_horizon[self.task_name]]
        self.target_flag_column_names = [i.replace('val', 'flag') for i in self.target_val_column_names]

        y_t = data[self.history_column_names].to_numpy()
        if self.external_flag:
            external_features = data[self.config['exp_params']['external_features']].to_numpy()

        history_y_t = list()
        for index in range(self.sliding_window, -1, -1):
            history_y_t.append(y_t[training_validation_index-index])
            if self.external_flag:
                history_y_t.append(external_features[training_validation_index-index])
        history_y_t = np.concatenate(history_y_t, axis=-1)

        training_validation_target_val = data[self.target_val_column_names].to_numpy()[training_validation_index[self.sliding_window:]]
        training_validation_target_flag = data[self.target_flag_column_names].to_numpy()[training_validation_index[self.sliding_window:]]
        selected_index = np.argwhere(np.prod(training_validation_target_flag, axis=-1) == 1).reshape([-1])
        num_train = int(len(selected_index) * self.config['exp_params']['train_valid_ratio'])

        train_x, train_y = history_y_t[selected_index[:num_train]], training_validation_target_val[selected_index[:num_train]]
        valid_x, valid_y = history_y_t[selected_index[num_train:]], training_validation_target_val[selected_index[num_train:]]
        if 'normalization' in self.param_dict and self.param_dict['normalization'] != 'none':
            if self.param_dict['normalization'] == 'minmax':
                self.scalar_x, self.scalar_y = MinMaxScaler(), MinMaxScaler()
            elif self.param_dict['normalization'] == 'standard':
                self.scalar_x, self.scalar_y = StandardScaler(), StandardScaler()
            self.scalar_x = self.scalar_x.fit(history_y_t)
            self.scalar_y = self.scalar_y.fit(training_validation_target_val)
            train_x, valid_x = self.scalar_x.transform(train_x), self.scalar_x.transform(valid_x)
            train_y, valid_y = self.scalar_y.transform(train_y), self.scalar_y.transform(valid_y)
        self.train_dataset = HistoryConcatTrainDataset(
            torch.from_numpy(train_x).to(torch.float),
            torch.from_numpy(train_y).to(torch.float),
            torch.from_numpy(training_validation_target_flag[selected_index[:num_train]]).to(torch.float))
        self.valid_dataset = HistoryConcatTrainDataset(
            torch.from_numpy(valid_x).to(torch.float),
            torch.from_numpy(valid_y).to(torch.float),
            torch.from_numpy(training_validation_target_flag[selected_index[num_train:]]).to(torch.float))

        """
        prepare testing datasets
        """
        testing_index = np.sort(np.argwhere(train_flag == 0).reshape([-1]))
        testing_data = data.iloc[testing_index]
        testing_ID = testing_data['ID'].to_numpy()
        history_y_t = list()
        for index in range(self.sliding_window, -1, -1):
            history_y_t.append(y_t[testing_index-index])
            if self.external_flag:
                history_y_t.append(external_features[testing_index-index])
        history_y_t = np.concatenate(history_y_t, axis=-1)

        if 'normalization' in self.param_dict and self.param_dict['normalization'] != 'none':
            history_y_t = self.scalar_x.transform(history_y_t)

        self.test_dataset = HistoryConcatTestDataset(
            torch.from_numpy(testing_ID).to(torch.int),
            torch.from_numpy(history_y_t).to(torch.float))

        return

    def load_train(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    def load_valid(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, drop_last=False)

    def load_test(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, drop_last=False)

class RegressorNet(nn.Module):
    def __init__(self,
                 sliding_window, external_features, history_column_names, target_column_names,
                 hidden_size, num_layers, dropout, normalization):
        super().__init__()

        self.normalization = normalization

        modules = list()
        last_dims = (sliding_window + 1) * (len(external_features) + len(history_column_names))
        for l in range(num_layers):
            modules.append(
                nn.Sequential(
                    nn.Linear(last_dims, hidden_size),
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropout)
                )
            )
            last_dims = hidden_size
        modules.append(nn.Linear(last_dims, len(target_column_names)))
        self.fcs = nn.Sequential(*modules)

    def forward(self, x):
        pred = self.fcs(x)
        if self.normalization == 'minmax':
            pred = torch.sigmoid(pred)
        return pred

    def loss_function(self, batch):
        x, y, flag = batch
        pred = self.forward(x)
        if self.normalization == 'none':
            loss = torch.mean(nn.MSELoss(reduction='none')(pred, torch.log(y)) * flag)
            pred = torch.exp(pred)
        else:
            loss = torch.mean(nn.MSELoss(reduction='none')(pred, y) * flag)
        return loss, pred

class FNN_exp(Pytorch_DNN_exp):
    def __init__(self, file, param_dict, config):
        super().__init__(file, param_dict, config)

        self.dataloader = HistoryConcatLoader(
            file,
            param_dict,
            config
        )
        self.model = self.load_model()

        print_network(self.model)
        """
        7.2 K     Trainable params
        0         Non-trainable params
        7.2 K     Total params
        0.029     Total estimated model params size (MB)
        """

    def load_model(self):
        model = RegressorNet(
            sliding_window=self.param_dict['sliding_window'],
            external_features=self.config['exp_params']['external_features'],
            history_column_names=self.dataloader.history_column_names,
            target_column_names=self.dataloader.target_val_column_names,
            hidden_size=self.param_dict['hidden_size'],
            num_layers=self.param_dict['num_layers'],
            dropout=self.param_dict['dropout'],
            normalization=self.param_dict['normalization']
        )
        return model

def grid_search_FNN(config, num_files):
    # set random seed
    torch.manual_seed(config['logging_params']['manual_seed'])
    torch.cuda.manual_seed(config['logging_params']['manual_seed'])
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
        'hidden_size': config['model_params']['hidden_size'],
        'num_layers': config['model_params']['num_layers'],
        'batch_size': config['exp_params']['batch_size'],
        'learning_rate': config['exp_params']['learning_rate'],
        'dropout': config['model_params']['dropout'],
        'normalization': config['exp_params']['normalization'],
    }
    param_dict_list = list(ParameterGrid(param_grid))
    """
    getting validation results
    """

    for file in file_list:
        cur_log_dir = os.path.join(log_dir, file.split('.')[0])
        if not config['exp_params']['test_flag']:
            if not os.path.exists(cur_log_dir):
                os.makedirs(cur_log_dir)
            Pytorch_DNN_validation(os.path.join(data_folder, file), param_dict_list, cur_log_dir, config, FNN_exp)
            """
            hyperparameters selection
            """
            summary = OrderedDict()
            for param_index, param_dict in enumerate(param_dict_list):
                param_dict = OrderedDict(param_dict)
                setting_name = 'param'
                for key, val in param_dict.items():
                    setting_name += f'_{key[0].capitalize()}{val}'

                model_list = [i for i in os.listdir(os.path.join(cur_log_dir, setting_name, 'version_0')) if i.endswith('.ckpt')]
                assert len(model_list) == 1
                perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('.ckpt')])
                with open(os.path.join(cur_log_dir, setting_name, 'version_0', 'std.txt'), 'r') as f:
                    std_text = f.readlines()
                    std_list = [[int(i.split()[0]), list(map(float, i.split()[1].split('_')))] for i in std_text]
                    std_dict = dict(zip(list(zip(*std_list))[0], list(zip(*std_list))[1]))
                best_epoch = int(model_list[0][model_list[0].find('best-epoch=')+len('best-epoch='):model_list[0].find('-avg_val_metric')])
                std = std_dict[best_epoch]
                # perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('-std')])
                # std = float(model_list[0][model_list[0].find('-std=')+len('-std='):model_list[0].find('.ckpt')])
                summary['_'.join(map(str, list(param_dict.values())))] = [perf, std]
            with open(os.path.join(cur_log_dir, 'val_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)

            selected_index = np.argmin(np.array(list(summary.values()))[:, 0])
            selected_params = list(summary.keys())[selected_index]
            param_dict = {
                'batch_size': int(selected_params.split('_')[0]),
                'dropout': float(selected_params.split('_')[1]),
                'hidden_size': int(selected_params.split('_')[2]),
                'learning_rate': float(selected_params.split('_')[3]),
                'normalization': selected_params.split('_')[4],
                'num_layers': int(selected_params.split('_')[5]),
                'sliding_window': int(selected_params.split('_')[6]),
                'std': np.array(list(summary.values()))[selected_index][-1],
            }
            # save param
            with open(os.path.join(cur_log_dir, 'param.json'), 'w') as f:
                json.dump(param_dict, f, indent=4)

        """
        prediction on testing
        """
        with open(os.path.join(cur_log_dir, 'param.json'), 'r') as f:
            param_dict = json.load(f)
        Pytorch_DNN_testing(os.path.join(data_folder, file), param_dict, cur_log_dir, config, FNN_exp)

    if not os.path.exists(os.path.join(log_dir, 'config.yaml')):
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

    parser.add_argument('--sliding_window', '-sliding_window', type=str, help='list of sliding_window for arima')
    parser.add_argument('--selection_metric', '-selection_metric', type=str, help='metrics to select hyperparameters, one of [RMSE, MAE, MAPE]',)
    parser.add_argument('--train_valid_ratio', '-train_valid_ratio', type=float, help='select hyperparameters on validation set')
    parser.add_argument('--external_features', '-external_features', type=str, help='list of external feature name list')

    # model-specific features
    parser.add_argument('--hidden_size', '-hidden_size', type=str, help='list of hidden_size')
    parser.add_argument('--num_layers', '-num_layers', type=str, help='list of num_layers')
    parser.add_argument('--batch_size', '-batch_size', type=str, help='list of batch_size')
    parser.add_argument('--max_epochs', '-max_epochs', type=int, help='number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type=int, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str)#, default='[1]')
    parser.add_argument('--dropout', '-dropout', type=str, help='list of dropout rates')

    args = vars(parser.parse_args())
    with open('./../configs/FNN.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    grid_search_FNN(config, num_files=args['num_files'])

