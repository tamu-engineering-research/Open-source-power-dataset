# Created by xunannancy at 2021/9/21
"""
conv1d
"""
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import math
import torch
from utils import seqlen, num_features, print_network, Pytorch_DNN_validation, Pytorch_DNN_testing, merge_parameters, \
    run_evaluate
from FNN import FNN_exp
import numpy as np
import os
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
from collections import OrderedDict
import json
import yaml
import argparse

class CNNNet(nn.Module):
    def __init__(self,
                 dropout, hidden_layers, kernel_size, stride, pooling, num_target_label
                 ):
        super().__init__()

        modules = list()
        last_channels = num_features
        padding, dilation = 0, 1
        L_in = seqlen
        for cur_out_channels, cur_kernel_size, cur_stride in zip(hidden_layers, kernel_size, stride):
            cnn_L_out = math.floor(
                (L_in + 2 * padding - dilation * (cur_kernel_size - 1) - 1)/cur_stride + 1
            )
            if pooling == 'max':
                cur_pooling = nn.MaxPool1d(
                    kernel_size=cur_kernel_size,
                    stride=cur_stride)
                pool_L_out = math.floor(
                    (cnn_L_out + 2 * padding - dilation * (cur_kernel_size - 1) - 1)/cur_stride + 1
                )
            elif pooling == 'avg':
                cur_pooling = nn.AvgPool1d(
                    kernel_size=cur_kernel_size,
                    stride=cur_stride
                )
                pool_L_out = math.floor(
                    (cnn_L_out + 2 * padding - cur_kernel_size) / cur_stride + 1
                )
            elif pooling == 'AdaptiveAvg':
                cur_pooling = nn.AdaptiveAvgPool1d(output_size=cnn_L_out)
                pool_L_out = cnn_L_out
            elif pooling == 'AdaptiveMax':
                cur_pooling = nn.AdaptiveMaxPool1d(output_size=cnn_L_out)
                pool_L_out = cnn_L_out
            elif pooling == 'lp':
                cur_pooling = nn.LPPool1d(
                    norm_type=2,
                    kernel_size=cur_kernel_size,
                    stride=cur_stride
                )
                pool_L_out = math.floor(
                    (cnn_L_out - cur_kernel_size) / cur_stride + 1
                )

            modules.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=last_channels,
                    out_channels=cur_out_channels,
                    kernel_size=cur_kernel_size,
                    stride=cur_stride
                ),
                cur_pooling,
                nn.LeakyReLU(),
                nn.Dropout(p=dropout)
            ))
            L_in = pool_L_out
            last_channels = cur_out_channels
        self.cnn_layer = nn.Sequential(*modules)
        # [N, Cout, Lout] => [N, Cout*Lout]
        self.final_layer = nn.Sequential(
            nn.Linear(cur_out_channels * pool_L_out, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_target_label),
        )

    def forward(self, x):
        N, C_in, L_in = x.shape[0], num_features, seqlen
        x_reshape = x.reshape([N, L_in, C_in]).permute([0, 2, 1])
        cnn_results = self.cnn_layer(x_reshape).reshape([N, -1])
        pred = self.final_layer(cnn_results)
        return pred

    def loss_function(self, batch):
        x, y = batch
        pred = self.forward(x)
        loss = nn.CrossEntropyLoss()(pred, y)
        return loss, torch.argmax(pred, dim=-1)

class CNN_exp(FNN_exp):
    def __init__(self, data_path, param_dict, config):
        super().__init__(data_path, param_dict, config)

    def load_model(self):
        model = CNNNet(
            dropout=self.param_dict['dropout'],
            hidden_layers=self.param_dict['hidden_layers'],
            kernel_size=self.param_dict['kernel_size'],
            stride=self.param_dict['stride'],
            pooling=self.param_dict['pooling'],
            num_target_label=len(self.dataloader.new_true_label_mapping)
        )
        print_network(model)
        return model

def grid_search_cnn(config):
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

    data_path = config['exp_params']['data_path']
    param_grid = {
        'dropout': config['model_params']['dropout'],
        'hidden_layers': config['model_params']['hidden_layers'],
        'kernel_size': config['model_params']['kernel_size'],
        'stride': config['model_params']['stride'],
        'pooling': config['model_params']['pooling'],

        'batch_size': config['exp_params']['batch_size'],
        'learning_rate': config['exp_params']['learning_rate'],
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
        Pytorch_DNN_validation(data_path, param_dict_list, log_dir, config, CNN_exp)

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
            'dropout': float(selected_params.split('_')[1]),
            'hidden_layers': eval(selected_params.split('_')[2]),
            'kernel_size': eval(selected_params.split('_')[3]),
            'label_constraints': eval(selected_params.split('_')[4]),
            'learning_rate': float(selected_params.split('_')[5]),
            'normalization': selected_params.split('_')[6],
            'pooling': selected_params.split('_')[7],
            'stride': eval(selected_params.split('_')[8]),
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
    Pytorch_DNN_testing(data_path, param_dict, log_dir, config, CNN_exp)

    if not os.path.exists(os.path.join(log_dir, 'config.yaml')):
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    evaluate_config = {
        'exp_params': {
            'prediction_path': log_dir,
        }
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

    parser.add_argument('--hidden_layers', '-hidden_layers', type=str, help='list of hidden layer options')
    parser.add_argument('--kernel_size', '-kernel_size', type=str, help='list of layerwise kernel options')
    parser.add_argument('--stride', '-stride', type=str, help='list of layerwise stride options')
    parser.add_argument('--pooling', '-pooling', type=str, help='list of pooling options')

    args = vars(parser.parse_args())
    with open('../configs/cnn.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    grid_search_cnn(config)

    """
    std:
    (drop0.) summary: {'#samples': 110, 'fault': 0.5154183412495433, 'location': -1, 'starttime': -1}    
    (drop0.1) summary: {'#samples': 110, 'fault': 0.5668615272195835, 'location': -1, 'starttime': -1}
    summary: {'#samples': 110, 'fault': -1, 'location': 0.14610917537746806, 'starttime': -1}    
    summary: {'#samples': 110, 'fault': -1, 'location': -1, 'starttime': 43.6849593495935}
    
    minmax:
    (drop0.1) summary: {'#samples': 110, 'fault': 0.5019762845849802, 'location': -1, 'starttime': -1}    
    """
