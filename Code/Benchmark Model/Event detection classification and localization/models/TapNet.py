# Created by xunannancy at 2021/9/21
"""
adopt implementation from https://github.com/xuczhang/tapnet.git
"""
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from utils import num_features, seqlen, Pytorch_DNN_validation, Pytorch_DNN_testing, print_network, compute_MMAE, \
    merge_parameters, run_evaluate
from sklearn.metrics import balanced_accuracy_score
from torch import optim
from FNN import FNN_exp
from collections import OrderedDict
from sklearn.model_selection import ParameterGrid
from copy import deepcopy
import json
import yaml
import argparse
import math

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def output_conv_size(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1
    return output


class TapNet_model(nn.Module):
    def __init__(self,
                 num_target_label,
                 metric_param, dropout, filters, kernels, dilation, layers, use_rp, rp_params,
                 use_att=True, use_metric=False, use_lstm=False, use_cnn=True, lstm_dim=128):
        super().__init__()
        self.nclass = num_target_label
        self.metric_param = metric_param
        self.dropout = dropout
        self.use_metric = use_metric
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn

        # parameters for random projection
        self.use_rp = use_rp
        if rp_params[0] < 0:
            dim = num_features
            rp_params = [3, math.floor(dim / (3 / 2))]
        else:
            dim = num_features
            rp_params[1] = math.floor(dim / rp_params[1])

        rp_params = [int(l) for l in rp_params]

        self.rp_group, self.rp_dim = rp_params

        # LSTM
        self.channel = num_features
        self.ts_length = seqlen

        self.lstm_dim = lstm_dim
        self.lstm = nn.LSTM(self.ts_length, self.lstm_dim)

        paddings = [0, 0, 0]
        if self.use_rp:
            self.conv_1_models = nn.ModuleList()
            self.idx = []
            for i in range(self.rp_group):
                self.conv_1_models.append(nn.Conv1d(self.rp_dim, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1, padding=paddings[0]))
                self.idx.append(np.random.permutation(num_features)[0: self.rp_dim])
        else:
            self.conv_1 = nn.Conv1d(self.channel, filters[0], kernel_size=kernels[0], dilation=dilation, stride=1, padding=paddings[0])

        self.conv_bn_1 = nn.BatchNorm1d(filters[0])

        self.conv_2 = nn.Conv1d(filters[0], filters[1], kernel_size=kernels[1], stride=1, padding=paddings[1])

        self.conv_bn_2 = nn.BatchNorm1d(filters[1])

        self.conv_3 = nn.Conv1d(filters[1], filters[2], kernel_size=kernels[2], stride=1, padding=paddings[2])

        self.conv_bn_3 = nn.BatchNorm1d(filters[2])

        # compute the size of input for fully connected layers
        fc_input = 0
        if self.use_cnn:
            conv_size = seqlen
            for i in range(len(filters)):
                conv_size = output_conv_size(conv_size, kernels[i], 1, paddings[i])
            fc_input += conv_size
            #* filters[-1]
        if self.use_lstm:
            fc_input += conv_size * self.lstm_dim

        if self.use_rp:
            fc_input = self.rp_group * filters[2] + self.lstm_dim

        # Representation mapping function
        layers = [fc_input] + layers
        print("Layers", layers)
        self.mapping = nn.Sequential()
        for i in range(len(layers) - 2):
            self.mapping.add_module("fc_" + str(i), nn.Linear(layers[i], layers[i + 1]))
            self.mapping.add_module("bn_" + str(i), nn.BatchNorm1d(layers[i + 1]))
            self.mapping.add_module("relu_" + str(i), nn.LeakyReLU())

        # add last layer
        self.mapping.add_module("fc_" + str(len(layers) - 2), nn.Linear(layers[-2], layers[-1]))
        if len(layers) == 2:  # if only one layer, add batch normalization
            self.mapping.add_module("bn_" + str(len(layers) - 2), nn.BatchNorm1d(layers[-1]))

        # Attention
        att_dim, semi_att_dim = 128, 128
        self.use_att = use_att
        if self.use_att:
            self.att_models = nn.ModuleList()
            for _ in range(num_target_label):
                att_model = nn.Sequential(
                    nn.Linear(layers[-1], att_dim),
                    nn.Tanh(),
                    nn.Linear(att_dim, 1)
                )
                self.att_models.append(att_model)

    def embedding(self, x):
        N = x.size(0)

        # LSTM
        if self.use_lstm:
            x_lstm = self.lstm(x)[0]
            x_lstm = x_lstm.mean(1)
            x_lstm = x_lstm.view(N, -1)

        if self.use_cnn:
            # Covolutional Network
            # input ts: # N * C * L
            if self.use_rp:
                for i in range(len(self.conv_1_models)):
                    #x_conv = x
                    x_conv = self.conv_1_models[i](x[:, self.idx[i], :])
                    x_conv = self.conv_bn_1(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_2(x_conv)
                    x_conv = self.conv_bn_2(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = self.conv_3(x_conv)
                    x_conv = self.conv_bn_3(x_conv)
                    x_conv = F.leaky_relu(x_conv)

                    x_conv = torch.mean(x_conv, 2)

                    if i == 0:
                        x_conv_sum = x_conv
                    else:
                        x_conv_sum = torch.cat([x_conv_sum, x_conv], dim=1)

                x_conv = x_conv_sum
            else:
                x_conv = x
                x_conv = self.conv_1(x_conv)  # N * C * L
                x_conv = self.conv_bn_1(x_conv)
                x_conv = F.leaky_relu(x_conv)

                x_conv = self.conv_2(x_conv)
                x_conv = self.conv_bn_2(x_conv)
                x_conv = F.leaky_relu(x_conv)

                x_conv = self.conv_3(x_conv)
                x_conv = self.conv_bn_3(x_conv)
                x_conv = F.leaky_relu(x_conv)

                x_conv = x_conv.view(N, -1)

        if self.use_lstm and self.use_cnn:
            x = torch.cat([x_conv, x_lstm], dim=1)
        elif self.use_lstm:
            x = x_lstm
        elif self.use_cnn:
            x = x_conv
        #

        # linear mapping to low-dimensional space
        x = self.mapping(x)

        return x

    def forward(self, x_train, y_train, x):
        """
        :param x_train: [batch_size, feature_dim, seqlen]
        :param y_train: [batch_size]
        :param x_test
        :return:
        """
        x, x_train = x.permute([0, 2, 1]), x_train.permute([0, 2, 1])
        x_train, y_train = x_train.to(x.device), y_train.to(x.device)

        x = self.embedding(x)
        x_train = self.embedding(x_train)

        # generate the class protocal with dimension C * D (nclass * dim)
        proto_list = []
        for i in range(self.nclass):
            idx = (y_train == i).nonzero().squeeze(1)
            if self.use_att:
                A = self.att_models[i](x_train[idx])  # N_k * 1
                A = torch.transpose(A, 1, 0)  # 1 * N_k
                A = F.softmax(A, dim=1)  # softmax over N_k

                class_repr = torch.mm(A, x_train[idx]) # 1 * L
                class_repr = torch.transpose(class_repr, 1, 0)  # L * 1
            else:  # if do not use attention, simply use the mean of training samples with the same labels.
                class_repr = x_train[idx].mean(0)  # L * 1
            proto_list.append(class_repr.view(1, -1))
        x_proto = torch.cat(proto_list, dim=0)

        # prototype distance
        proto_dists = euclidean_dist(x_proto, x_proto)
        proto_dists = torch.exp(-0.5*proto_dists)
        num_proto_pairs = int(self.nclass * (self.nclass - 1) / 2)
        proto_dist = torch.sum(proto_dists) / num_proto_pairs

        dists = euclidean_dist(x, x_proto)

        return torch.exp(-0.5*dists), proto_dist

    def loss_function(self, x_train, y_train, x, y):
        pred, proto_dist = self.forward(x_train, y_train, x)
        loss = nn.CrossEntropyLoss()(pred, y)
        if self.use_metric:
            loss += self.metric_param * proto_dist
        return loss, torch.argmax(pred, dim=-1)

class TapNet_exp(FNN_exp):
    def __init__(self, data_path, param_dict, config):
        super().__init__(data_path, param_dict, config)
        self.x_train, self.y_train = self.dataloader.train_dataset.x, self.dataloader.train_dataset.y

    def load_model(self):
        model = TapNet_model(
            num_target_label=len(self.dataloader.new_true_label_mapping),
            metric_param=self.param_dict['metric_param'],
            dropout=self.param_dict['dropout'],
            filters=self.param_dict['filters'],
            kernels=self.param_dict['kernels'],
            dilation=self.param_dict['dilation'],
            layers=self.param_dict['layers'],
            use_rp=self.param_dict['use_rp'],
            rp_params=self.param_dict['rp_params'],
            use_metric=self.param_dict['use_metric'],
            use_lstm=self.param_dict['use_lstm'],
            use_cnn=self.param_dict['use_cnn'],
            lstm_dim=self.param_dict['lstm_dim']
        )
        print_network(model)
        return model

    def oracle_loss(self, batch):
        x, y = batch
        loss, pred = self.model.loss_function(self.x_train, self.y_train, x, y)
        if self.target_name in ['fault', 'location']:
            # balanced_acc
            perf = balanced_accuracy_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        elif self.target_name == 'starttime':
            perf = compute_MMAE(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        return {'loss': loss, 'metric': perf, 'pred': pred, 'y': batch[1]}

    def test_step(self, batch, batch_idx):
        x = batch
        pred, _ = self.model.forward(self.x_train, self.y_train, x)
        return torch.argmax(pred, dim=-1)

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.param_dict['learning_rate'],
            weight_decay=self.param_dict['weight_decay']
        )
        optims.append(optimizer)
        return optims, scheds


def grid_search_TapNet(config):
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
        'batch_size': config['exp_params']['batch_size'],
        'learning_rate': config['exp_params']['learning_rate'],
        'weight_decay': config['exp_params']['weight_decay'],
        'dropout': config['model_params']['dropout'],
        'normalization': config['exp_params']['normalization'],
        'target_name': config['exp_params']['target_name'],
        'label_constraints': config['exp_params']['label_constraints'],

        'use_cnn': config['model_params']['use_cnn'],
        'filters': config['model_params']['filters'],
        'kernels': config['model_params']['kernels'],
        'dilation': config['model_params']['dilation'],
        'layers': config['model_params']['layers'],
        'use_lstm': config['model_params']['use_lstm'],
        'lstm_dim': config['model_params']['lstm_dim'],
        'use_rp': config['model_params']['use_rp'],
        'rp_params': config['model_params']['rp_params'],
        'use_metric': config['model_params']['use_metric'],
        'metric_param': config['model_params']['metric_param'],
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
        Pytorch_DNN_validation(data_path, param_dict_list, log_dir, config, TapNet_exp)

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
            'dilation': int(selected_params.split('_')[1]),
            'dropout': float(selected_params.split('_')[2]),
            'filters': eval(selected_params.split('_')[3]),
            'kernels': eval(selected_params.split('_')[4]),
            'label_constraints': eval(selected_params.split('_')[5]),
            'layers': eval(selected_params.split('_')[6]),
            'learning_rate': float(selected_params.split('_')[7]),
            'lstm_dim': int(selected_params.split('_')[8]),
            'metric_param': float(selected_params.split('_')[9]),
            'normalization': selected_params.split('_')[10],
            'rp_params': eval(selected_params.split('_')[11]),
            'use_cnn': eval(selected_params.split('_')[12]),
            'use_lstm': eval(selected_params.split('_')[13]),
            'use_metric': eval(selected_params.split('_')[14]),
            'use_rp': eval(selected_params.split('_')[15]),
            'weight_decay': float(selected_params.split('_')[16]),
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
    Pytorch_DNN_testing(data_path, param_dict, log_dir, config, TapNet_exp)

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
    parser.add_argument('--weight_decay', '-weight_decay', type=int, help='list of weight decay options')
    parser.add_argument('--gpus', '-g', type=str, help='list of available gpus')
    parser.add_argument('--dropout', '-dropout', type=str, help='list of dropout rates')

    parser.add_argument('--label_constraints', '-label_constraints', type=str, help='list of optional label constraints')
    parser.add_argument('--target_name', '-target_name', type=str, help='subtasks to complete')

    parser.add_argument('--use_cnn', '-use_cnn', type=str, help='list of use_cnn options')
    parser.add_argument('--filters', '-filters', type=str, help='list of filters options')
    parser.add_argument('--kernels', '-kernels', type=str, help='list of kernels options')
    parser.add_argument('--dilation', '-dilation', type=str, help='list of dilation options')
    parser.add_argument('--layers', '-layers', type=str, help='list of layers options')
    parser.add_argument('--use_lstm', '-use_lstm', type=str, help='list of use_lstm options')
    parser.add_argument('--lstm_dim', '-lstm_dim', type=str, help='list of lstm_dim options')
    parser.add_argument('--use_rp', '-use_rp', type=str, help='list of use_rp options')
    parser.add_argument('--rp_params', '-rp_params', type=str, help='list of rp_params options')
    parser.add_argument('--use_metric', '-use_metric', type=str, help='list of use_metric options')
    parser.add_argument('--metric_param', '-metric_param', type=str, help='list of metric_param options')

    args = vars(parser.parse_args())
    with open('./../configs/TapNet.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')


    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    grid_search_TapNet(config)




