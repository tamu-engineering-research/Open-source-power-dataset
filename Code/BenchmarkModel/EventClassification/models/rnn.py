# Created by xunannancy at 2021/9/21
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch
from FNN import FNN_exp
import os
import numpy as np
import json
import yaml
from collections import OrderedDict
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
from utils import Pytorch_DNN_validation, Pytorch_DNN_testing, merge_parameters, num_features, run_evaluate
import argparse

class RNNNet(nn.Module):
    def __init__(self,
                 hidden_size, num_layers, dropout, direction, model_name, num_target_label):
        super().__init__()

        self.rnn_cell = model_name.upper()

        assert self.rnn_cell in ['RNN', 'LSTM', 'GRU']
        assert direction in ['uni', 'bi']
        self.direction = direction
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout)
        )

        if self.direction == 'uni':
            bidirectional = False
            self.cur_rnn_dim = self.hidden_size
        elif self.direction == 'bi':
            bidirectional = True
            self.cur_rnn_dim = int(self.hidden_size/2)

        self.rnn_layer = eval(f'nn.{self.rnn_cell}')(
            input_size=self.hidden_size,
            hidden_size=self.cur_rnn_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_target_label)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        embed = self.embedding(x)
        rnn_results = self.rnn_layer(embed)
        if self.rnn_cell in ['RNN', 'GRU']:
            hidden = rnn_results[1]
        elif self.rnn_cell == 'LSTM':
            hidden = rnn_results[1][0]
        """
        RNN: output, h_n
        GRU: output, h_n
        LSTM: output, (h_n, c_n)
        """
        pred = self.final_layer(hidden.reshape([(self.direction=='bi')+1, self.num_layers, batch_size, self.cur_rnn_dim])[:, -1, :].permute([1, 0, 2]).reshape([batch_size, self.hidden_size]))
        return pred

    def loss_function(self, batch):
        x, y = batch
        pred = self.forward(x)
        loss = nn.CrossEntropyLoss()(pred, y)
        return loss, torch.argmax(pred, dim=-1)

class RNN_exp(FNN_exp):
    def __init__(self, data_path, param_dict, config):
        super().__init__(data_path, param_dict, config)

    def load_model(self):
        model = RNNNet(
            hidden_size=self.param_dict['hidden_size'],
            num_layers=self.param_dict['num_layers'],
            dropout=self.param_dict['dropout'],
            direction=self.param_dict['direction'],
            model_name=self.config['model_params']['model_name'],
            num_target_label=len(self.dataloader.new_true_label_mapping)
        )
        return model

def grid_search_RNN(config):
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

    data_path = config['exp_params']['data_path']

    param_grid = {
        'hidden_size': config['model_params']['hidden_size'],
        'num_layers': config['model_params']['num_layers'],
        'batch_size': config['exp_params']['batch_size'],
        'learning_rate': config['exp_params']['learning_rate'],
        'dropout': config['model_params']['dropout'],
        'direction': config['model_params']['direction'],
        'normalization': config['exp_params']['normalization'],
        'target_name': config['exp_params']['target_name'],
        'label_constraints': config['exp_params']['label_constraints'],
    }
    origin_param_dict_list = list(ParameterGrid(param_grid))
    param_dict_list, param_dict_nick = list(), list()
    # remove label_constraints=True/False for fault type
    for param_index, param_dict in enumerate(origin_param_dict_list):
        if param_dict['target_name'] == 'fault':
            tmp = deepcopy(param_dict)
            del tmp['label_constraints']
            cur_nick = '_'.join(map(str, (OrderedDict(tmp).values())))
            if cur_nick in param_dict_nick:
                continue
            param_dict_nick.append(cur_nick)
        param_dict_list.append(origin_param_dict_list[param_index])

    """
    getting validation results
    """
    if not config['exp_params']['test_flag']:
        Pytorch_DNN_validation(data_path, param_dict_list, log_dir, config, RNN_exp)

    """
    hyperparameters selection
    """
    summary, param_dict_res = OrderedDict(), dict()
    for target_name in config['exp_params']['target_name']:
        summary[target_name] = OrderedDict()
        for param_index, param_dict in enumerate(param_dict_list):
            if param_dict['target_name'] != target_name:
                continue
            param_dict = OrderedDict(param_dict)
            setting_name = target_name
            for key, val in param_dict.items():
                if key == 'target_name':
                    continue
                setting_name += f'_{key[0].capitalize()}{val}'
            model_list = [i for i in os.listdir(os.path.join(log_dir, setting_name, 'version_0')) if i.endswith('.ckpt')]
            assert len(model_list) == 1
            perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('.ckpt')])
            summary[target_name]['_'.join(map(str, [j for i, j in param_dict.items() if i!='target_name']))] = perf

        reference = np.array(list(summary[target_name].values()))
        if target_name in ['fault', 'location']:
            selected_index = np.argmax(reference)
        elif target_name == 'starttime':
            selected_index = np.argmin(reference)
        selected_params = list(summary[target_name].keys())[selected_index]
        param_dict_res[target_name] = {
            'batch_size': int(selected_params.split('_')[0]),
            'direction': selected_params.split('_')[1],
            'dropout': float(selected_params.split('_')[2]),
            'hidden_size': int(selected_params.split('_')[3]),
            'label_constraints': eval(selected_params.split('_')[4]),
            'learning_rate': float(selected_params.split('_')[5]),
            'normalization': selected_params.split('_')[6],
            'num_layers': int(selected_params.split('_')[7]),
        }
    with open(os.path.join(log_dir, 'val_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    with open(os.path.join(log_dir, 'param.json'), 'w') as f:
        json.dump(param_dict_res, f, indent=4)

    """
    prediction on testing
    """
    with open(os.path.join(log_dir, 'param.json'), 'r') as f:
        param_dict = json.load(f)
    Pytorch_DNN_testing(data_path, param_dict, log_dir, config, RNN_exp)

    if not os.path.exists(os.path.join(log_dir, 'config.yaml')):
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    evaluate_config = {
        'exp_params': {
            'prediction_path': log_dir,
            'data_path': config['exp_params']['data_path'],
        },
    }
    run_evaluate(config=evaluate_config, verbose=False)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--train_valid_ratio', '-train_valid_ratio', type=float, help='select hyperparameters on validation set')
    parser.add_argument('--manual_seed', '-manual_seed', type=int, help='manual_seed')

    # model-specific features
    parser.add_argument('--hidden_size', '-hidden_size', type=str, help='list of hidden_size')
    parser.add_argument('--num_layers', '-num_layers', type=str, help='list of num_layers')
    parser.add_argument('--batch_size', '-batch_size', type=str, help='list of batch_size')
    parser.add_argument('--max_epochs', '-max_epochs', type=int, help='number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type=int, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str)#, default='[1]')
    parser.add_argument('--dropout', '-dropout', type=str, help='list of dropout rates')

    parser.add_argument('--label_constraints', '-label_constraints', type=str, help='list of optional label constraints')
    parser.add_argument('--target_name', '-target_name', type=str, help='subtasks to complete')

    parser.add_argument('--direction', '-direction', type=str, help='list of direction options')
    parser.add_argument('--model_name', '-model_name', type=str, help='model_name, one of RNN, GRU, LSTM')

    args = vars(parser.parse_args())
    config_path = './../configs/RNN.yaml'
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')


    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    config['logging_params']['name'] = config['model_params']['model_name']
    grid_search_RNN(config)

    """
    GRU:
    std
    uni
    summary: {'#samples': 110, 'fault': 0.6430265386787125, 'location': -1, 'starttime': -1}
    summary: {'#samples': 110, 'fault': -1, 'location': 0.26277584204413473, 'starttime': -1}
    summary: {'#samples': 110, 'fault': -1, 'location': -1, 'starttime': 50.13617886178862}
    bi
    summary: {'#samples': 110, 'fault': 0.36470588235294116, 'location': 0.06953155245838173, 'starttime': 42.85772357723577}
    
    
    minmax:
    summary: {'#samples': 110, 'fault': 0.22727272727272724, 'location': -1, 'starttime': -1}
    
    
    RNN:
    std:
    summary: {'#samples': 110, 'fault': 0.46447337828412005, 'location': -1, 'starttime': -1}
    summary: {'#samples': 110, 'fault': -1, 'location': 0.18569492837785523, 'starttime': -1}
    
    LSTM:
    std:
    summary: {'#samples': 110, 'fault': 0.6230245457866942, 'location': 0.1765389082462253, 'starttime': 56.930894308943095}    
    summary: {'#samples': 110, 'fault': 0.5394492975055635, 'location': 0.17241579558652728, 'starttime': 57.28861788617886}    
    bi:
    summary: {'#samples': 110, 'fault': -1, 'location': -1, 'starttime': 53.426829268292686}
    
    """