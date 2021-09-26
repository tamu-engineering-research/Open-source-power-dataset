# Created by xunannancy at 2021/9/25
"""
refer to codes: https://github.com/ElementAI/N-BEATS.git
"""
import  warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch
from typing import Tuple
from utils import merge_parameters, Pytorch_DNN_exp, Pytorch_DNN_testing, Pytorch_DNN_validation, print_network, run_evaluate_V3
import numpy as np
from FNN import HistoryConcatLoader
import os
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
from collections import OrderedDict
import json
import yaml
import argparse

class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """
    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)

class GenericBasis(nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class NBeats(nn.Module):
    """
    N-Beats Model.
    """
    def __init__(self,
                 sliding_window, external_flag, external_features, history_column_names, target_val_column_names, normalization,
                 stacks, layers, layer_size,  location_type='multiple',
                 ):
        super().__init__()
        self.history_column_names = history_column_names
        self.target_val_column_names = target_val_column_names
        self.sliding_window = sliding_window
        self.external_flag, self.external_features = external_flag, external_features
        self.normalization = normalization

        # if self.location_type == 'single':
        #     input_size = output_size = 1
        # elif self.location_type == 'multiple':
        #     input_size = output_size = len(self.loc_index)
        # if self.external_flag:
        #     input_size += len(self.external_features)
        # input_size *= (sliding_window+1)
        # output_size *= len(self.prediction_horizon)
        input_size = len(self.history_column_names)
        if self.external_flag:
            input_size += len(self.external_features)
        input_size *= (sliding_window+1)
        output_size = len(self.target_val_column_names)

        modules = list()
        for i in range(stacks):
            modules.append(NBeatsBlock(
                input_size=input_size,
                theta_size=input_size + output_size,
                basis_function=GenericBasis(
                    backcast_size=input_size,
                    forecast_size=output_size),
                layers=layers,
                layer_size=layer_size))
        self.blocks = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        multiple locations: [batch_size, (sliding_window+1)*(loc_index+external_features)]
        single location: [batch_size, (sliding_window+1)*(1+external_features)]
        :param x: [batch_size, input_size]
        :return: [batch_size, output_size]
        """
        residuals = x.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast

        if self.normalization == 'minmax':
            forecast = torch.sigmoid(forecast)
        return forecast

    def loss_function(self, batch):
        x, y, flag = batch
        pred = self.forward(x)
        if self.normalization == 'none':
            loss = torch.mean(nn.MSELoss(reduction='none')(pred, torch.log(y)) * flag)
            pred = torch.exp(pred)
        else:
            loss = torch.mean(nn.MSELoss(reduction='none')(pred, y) * flag)
        return loss, pred

class NBeats_exp(Pytorch_DNN_exp):
    def __init__(self, file, param_dict, config):
        super().__init__(file, param_dict, config)

        self.dataloader = HistoryConcatLoader(
            file,
            param_dict,
            config
        )
        self.model = self.load_model()
        print_network(self.model)

    def load_model(self):
        model = NBeats(
            sliding_window=self.param_dict['sliding_window'],
            external_flag=self.param_dict['external_flag'],
            external_features=self.config['exp_params']['external_features'],
            history_column_names=self.dataloader.history_column_names,
            target_val_column_names=self.dataloader.target_val_column_names,
            normalization=self.param_dict['normalization'],
            stacks=self.param_dict['stacks'],
            layers=self.param_dict['layers'],
            layer_size=self.param_dict['layer_size']
        )
        return model

def grid_search_NBeats(config, num_files):
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
        'batch_size': config['exp_params']['batch_size'],
        'learning_rate': config['exp_params']['learning_rate'],
        'normalization': config['exp_params']['normalization'],

        'stacks': config['model_params']['stacks'],
        'layers': config['model_params']['layers'],
        'layer_size': config['model_params']['layer_size'],
        'external_flag': config['exp_params']['external_flag'],
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
            Pytorch_DNN_validation(os.path.join(data_folder, file), param_dict_list, cur_log_dir, config, NBeats_exp)
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
                'external_flag': eval(selected_params.split('_')[1]),
                'layer_size': int(selected_params.split('_')[2]),
                'layers': int(selected_params.split('_')[3]),
                'learning_rate': float(selected_params.split('_')[4]),
                'normalization': selected_params.split('_')[5],
                'sliding_window': int(selected_params.split('_')[6]),
                'stacks': int(selected_params.split('_')[7]),
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
        Pytorch_DNN_testing(os.path.join(data_folder, file), param_dict, cur_log_dir, config, NBeats_exp)


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
    parser.add_argument('--num_files', '-num_files', type=int, default=3, help='number of files to predict')

    parser.add_argument('--sliding_window', '-sliding_window', type=str, help='list of sliding_window for arima')
    parser.add_argument('--selection_metric', '-selection_metric', type=str, help='metrics to select hyperparameters, one of [RMSE, MAE, MAPE]',)
    parser.add_argument('--train_valid_ratio', '-train_valid_ratio', type=float, help='select hyperparameters on validation set')
    parser.add_argument('--external_features', '-external_features', type=str, help='list of external feature name list')

    # model-specific features
    parser.add_argument('--batch_size', '-batch_size', type=str, help='list of batch_size')
    parser.add_argument('--max_epochs', '-max_epochs', type=int, help='number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type=str, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str)#, default='[1]')

    parser.add_argument('--stacks', '-stacks', type=str, help='list of stacks options')
    parser.add_argument('--layers', '-layers', type=str, help='list of layers options')
    parser.add_argument('--layer_size', '-layer_size', type=str, help='list of layer_size options')
    parser.add_argument('--location_type', '-location_type', type=str, help='list of location_type options')
    parser.add_argument('--external_flag', '-external_flag', type=str, help='list of external_flag options')

    args = vars(parser.parse_args())
    with open('./../configs/NBeats.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    grid_search_NBeats(config, num_files=args['num_files'])










