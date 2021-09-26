# Created by xunannancy at 2021/9/25
"""
refer to https://github.com/laiguokun/LSTNet.git
"""
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import Pytorch_DNN_exp, Pytorch_DNN_testing, Pytorch_DNN_validation, print_network, merge_parameters, run_evaluate_V3
import yaml
import json
import os
from FNN import HistoryConcatLoader
import numpy as np
from collections import OrderedDict
from sklearn.model_selection import ParameterGrid
import argparse
from torch import optim

class LSTNetModel(nn.Module):
    def __init__(self,
                 sliding_window, external_features, history_column_names, target_val_column_names, dropout, normalization,
                 hidRNN, hidCNN, hidSkip, cnn_kernel, skip, highway_window
                 ):
        super().__init__()
        self.history_column_names = history_column_names
        self.sliding_window = sliding_window
        self.external_features = external_features
        self.target_val_column_names = target_val_column_names
        self.normalization = normalization
        self.prediction_horizon = int(len(self.target_val_column_names)/len(self.history_column_names))

        self.P = self.sliding_window + 1
        self.m = len(self.history_column_names) + len(self.external_features)
        self.hidR = hidRNN
        self.hidC = hidCNN
        self.hidS = hidSkip
        self.Ck = cnn_kernel
        self.skip = skip
        # NOTE: force to be integer, 121, 6, 24
        self.pt = int((self.P - self.Ck)/self.skip)
        self.hw = highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=dropout)
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, len(self.target_val_column_names))
        else:
            self.linear1 = nn.Linear(self.hidR, len(self.target_val_column_names))
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, self.prediction_horizon)
        if self.normalization == 'minmax':
            self.output = torch.sigmoid
        else:
            self.output = None

    def forward(self, x):
        batch_size = x.size(0)
        # [batch_size, (sliding_window+1)*(loc+feature)] => [batch_size, window, feature_dim]
        x = x.reshape([batch_size, self.sliding_window+1, len(self.history_column_names) + len(self.external_features)])

        #CNN
        c = x.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))


        #skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        # [batch, prediction_horizon*loc_len]
        res = self.linear1(r)

        #highway
        if (self.hw > 0):
            # [batch_size, window, feature_dim] => [batch, highway_window, loc_len]
            z = x[:, -self.hw:, :len(self.history_column_names)]
            # => [batch, loc_len, highway_window] => [batch * loc_len, highway_window]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            # => [batch * loc_len, prediction_horizon]
            z = self.highway(z)
            # => [batch, loc_len, prediction_horizon] => [batch, prediction_horizon, loc_len] => [batch, prediction_horizon * loc_len]
            z = z.reshape([batch_size, len(self.history_column_names), self.prediction_horizon]).permute([0, 2, 1]).reshape([batch_size, len(self.target_val_column_names)])
            res = res + z

        if (self.output):
            res = self.output(res)

        if self.normalization == 'minmax':
            res = torch.sigmoid(res)
        return res

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

class LSTNet_exp(Pytorch_DNN_exp):
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
        model = LSTNetModel(
            sliding_window=self.param_dict['sliding_window'],
            external_features=self.config['exp_params']['external_features'],
            dropout=self.param_dict['dropout'],
            normalization=self.param_dict['normalization'],
            history_column_names=self.dataloader.history_column_names,
            target_val_column_names=self.dataloader.target_val_column_names,
            hidRNN=self.param_dict['hidRNN'],
            hidCNN=self.param_dict['hidCNN'],
            hidSkip=self.param_dict['hidSkip'],
            cnn_kernel=self.param_dict['cnn_kernel'],
            skip=self.param_dict['skip'],
            highway_window=self.param_dict['highway_window']
        )
        return model

def grid_search_LSTNet(config, num_files):
    # set random seed
    torch.manual_seed(config['logging_params']['manual_seed'])
    torch.cuda.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    torch.backends.cudnn.enabled = False

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

        'hidRNN': config['model_params']['hidRNN'],
        'hidCNN': config['model_params']['hidCNN'],
        'hidSkip': config['model_params']['hidSkip'],
        'cnn_kernel': config['model_params']['cnn_kernel'],
        'skip': config['model_params']['skip'],
        'highway_window': config['model_params']['highway_window'],
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
            Pytorch_DNN_validation(os.path.join(data_folder, file), param_dict_list, cur_log_dir, config, LSTNet_exp)
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
                'cnn_kernel': int(selected_params.split('_')[1]),
                'dropout': float(selected_params.split('_')[2]),
                'hidCNN': int(selected_params.split('_')[3]),
                'hidRNN': int(selected_params.split('_')[4]),
                'hidSkip': int(selected_params.split('_')[5]),
                'highway_window': int(selected_params.split('_')[6]),
                'learning_rate': float(selected_params.split('_')[7]),
                'normalization': selected_params.split('_')[8],
                'skip': int(selected_params.split('_')[9]),
                'sliding_window': int(selected_params.split('_')[10]),
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
        Pytorch_DNN_testing(os.path.join(data_folder, file), param_dict, cur_log_dir, config, LSTNet_exp)


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
    parser.add_argument('--batch_size', '-batch_size', type=str, help='list of batch_size')
    parser.add_argument('--max_epochs', '-max_epochs', type=int, help='number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type=int, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str)#, help='gpus')

    parser.add_argument('--dropout', '-dropout', type=str, help='list of dropout for convnet')
    parser.add_argument('--normalization', '-normalization', type=str, help='list of normalization')
    parser.add_argument('--hidRNN', '-hidRNN', type=str, help='list of number of RNN hidden units options')
    parser.add_argument('--hidCNN', '-hidCNN', type=str, help='list of number of CNN hidden units options')
    parser.add_argument('--hidSkip', '-hidSkip', type=str, help='list of hidSkip options')
    parser.add_argument('--cnn_kernel', '-cnn_kernel', type=str, help='list of the kernel size of the CNN layers options')
    parser.add_argument('--skip', '-skip', type=str, help='list of skip options')
    parser.add_argument('--highway_window', '-highway_window', type=str, help='list of the window size of the highway component options')

    args = vars(parser.parse_args())
    with open('./../configs/LSTNet.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    grid_search_LSTNet(config, num_files=args['num_files'])