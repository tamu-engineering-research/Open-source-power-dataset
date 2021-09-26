# Created by xunannancy at 2021/9/25
"""
codes borrowed from https://github.com/locuslab/TCN.git
follow model from TCN.tcn.TemporalConvNet and TCN.TCN.word_cnn.model
"""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from utils import merge_parameters, Pytorch_DNN_exp, Pytorch_DNN_testing, Pytorch_DNN_validation, print_network, \
    run_evaluate_V3
from FNN import HistoryConcatLoader
import numpy as np
import os
import yaml
import json
from collections import OrderedDict
from sklearn.model_selection import ParameterGrid
import argparse
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self,
                 sliding_window, external_features, history_column_names, target_val_column_names, dropout, normalization, classification_token,
                 num_channels, kernel_size
                 ):
        super().__init__()
        self.history_column_names = history_column_names
        self.sliding_window = sliding_window
        self.external_features = external_features
        self.target_val_column_names = target_val_column_names
        self.normalization = normalization
        self.dropout = dropout
        self.classification_token = classification_token
        self.num_channels = num_channels

        self.tcn = TemporalConvNet(
            num_inputs=len(self.history_column_names)+len(self.external_features),
            num_channels=self.num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        if self.classification_token == 'all':
            decoder_channel_in = (sliding_window + 1) * self.num_channels[-1]
        else:
            decoder_channel_in = self.num_channels[-1]
        self.decoder = nn.Linear(decoder_channel_in, len(self.target_val_column_names))
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)
        return

    def forward(self, x):
        batch_size = x.shape[0]
        # [batch_size, (sliding_window+1)*(loc+feature)] => [batch_size, (sliding_window+1), (loc+feature)] => [batch_size,  (loc+feature), (sliding_window+1)]
        input = x.reshape([batch_size, self.sliding_window+1, len(self.history_column_names) + len(self.external_features)]).permute([0, 2, 1])
        tcn_outputs = self.tcn(input)
        if self.classification_token == 'first':
            tcn_outputs = tcn_outputs[:, :, 0]
        elif self.classification_token == 'last':
            tcn_outputs = tcn_outputs[:, :, -1]
        else:
            tcn_outputs = tcn_outputs.transpose(1, 2).reshape([batch_size, (self.sliding_window+1)*self.num_channels[-1]])
        y = self.decoder(tcn_outputs)
        if self.normalization == 'minmax':
            y = torch.sigmoid(y)
        return y

    def loss_function(self, batch):
        """
        same as loss_function defined in class Wave2Wave from WaveNet.py
        :param batch:
        :return:
        """
        x, y, flag = batch
        pred = self.forward(x)
        if self.normalization == 'none':
            loss = torch.mean(nn.MSELoss(reduction='none')(pred, torch.log(y)) * flag)
            pred = torch.exp(pred)
        else:
            loss = torch.mean(nn.MSELoss(reduction='none')(pred, y) * flag)

        return loss, pred

class TCN_exp(Pytorch_DNN_exp):
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
        model = TCNModel(
            sliding_window=self.param_dict['sliding_window'],
            external_features=self.config['exp_params']['external_features'],
            history_column_names=self.dataloader.history_column_names,
            target_val_column_names=self.dataloader.target_val_column_names,
            dropout=self.param_dict['dropout'],
            normalization=self.param_dict['normalization'],
            classification_token=self.param_dict['classification_token'],
            num_channels=[self.param_dict['nhid']] * self.param_dict['levels'],
            kernel_size=self.param_dict['kernel_size']
        )
        return model

def grid_search_TCN(config, num_files):
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
        'dropout': config['model_params']['dropout'],

        'nhid': config['model_params']['nhid'],
        'levels': config['model_params']['levels'],
        'kernel_size': config['model_params']['kernel_size'],
        'classification_token': config['model_params']['classification_token'],
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
            Pytorch_DNN_validation(os.path.join(data_folder, file), param_dict_list, cur_log_dir, config, TCN_exp)
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
                'classification_token': selected_params.split('_')[1],
                'dropout': float(selected_params.split('_')[2]),
                'kernel_size': int(selected_params.split('_')[3]),
                'learning_rate': float(selected_params.split('_')[4]),
                'levels': int(selected_params.split('_')[5]),
                'nhid': int(selected_params.split('_')[6]),
                'normalization': selected_params.split('_')[7],
                'sliding_window': int(selected_params.split('_')[8]),
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
        Pytorch_DNN_testing(os.path.join(data_folder, file), param_dict, cur_log_dir, config, TCN_exp)
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
    parser.add_argument('--learning_rate', '-learning_rate', type=int, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str)#, default='[1]')

    parser.add_argument('--nhid', '-nhid', type=str, help='list of hidden neurons')
    parser.add_argument('--levels', '-levels', type=str, help='list of hidden layers')
    parser.add_argument('--dropout', '-dropout', type=str, help='list of dropout for convnet')
    parser.add_argument('--normalization', '-normalization', type=str, help='list of normalization')
    parser.add_argument('--classification_token', '-classification_token', type=str, help='list of classification_token')
    parser.add_argument('--kernel_size', '-kernel_size', type=str, help='list of kernel_size')

    args = vars(parser.parse_args())
    with open('./../configs/tcn.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    grid_search_TCN(config, num_files=args['num_files'])
