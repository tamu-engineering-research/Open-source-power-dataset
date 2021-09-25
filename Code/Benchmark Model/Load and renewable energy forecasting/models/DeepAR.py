# Created by xunannancy at 2021/9/25
"""
refer to codes at https://github.com/zhykoties/TimeSeries.git
"""
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import torch
from utils import merge_parameters, Pytorch_DNN_exp, Pytorch_DNN_testing, Pytorch_DNN_validation, print_network, \
    prediction_interval_multiplier, task_prediction_horizon, run_evaluate_V3
from FNN import HistoryConcatLoader
from collections import OrderedDict
import os
import json
import yaml
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
import numpy as np
import argparse
import itertools
import pandas as pd

class DeepAR_model(nn.Module):
    def __init__(self,
                 sliding_window, external_flag, external_features, history_column_names, target_val_column_names, dropout, normalization,
                 hidden_dim, hidden_layers
                 ):
        '''
        We define a recurrent network that predicts the future values of a time-dependent variable based on
        past inputs and covariates.
        '''
        super().__init__()
        self.history_column_names, self.target_val_column_names = history_column_names, target_val_column_names
        self.sliding_window = sliding_window
        self.external_flag, self.external_features = external_flag, external_features
        self.normalization = normalization

        self.input_size = len(self.history_column_names)
        if self.external_flag:
            self.input_size += len(self.external_features)
        self.output_size = len(self.target_val_column_names)

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=hidden_dim,
                            num_layers=hidden_layers,
                            bias=True,
                            batch_first=False,
                            dropout=dropout)
        '''self.lstm = nn.LSTM(input_size=1 + params.cov_dim,
                            hidden_size=params.lstm_hidden_dim,
                            num_layers=params.lstm_layers,
                            bias=True,
                            batch_first=False,
                            dropout=params.lstm_dropout)'''
        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(hidden_dim * hidden_layers, self.output_size)
        self.distribution_presigma = nn.Linear(hidden_dim * hidden_layers, self.output_size)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
            x: ([1, batch_size, 1+cov_dim]) => [seqlen, batch_size, loc_index+external_features]
        Returns:
            mu ([batch_size, output_size]): estimated mean of z_t
            sigma ([batch_size, output_size]): estimated standard deviation of z_t
        '''
        # [batch_size, (sliding_window+1)*(loc_index+external_features)]
        batch_size = x.shape[0]
        # => [batch_size, (sliding_window+1), (loc_index+external_features)] => [(sliding_window+1), batch_size, (loc_index+external_features)]
        x = x.reshape([batch_size, self.sliding_window+1, self.input_size]).permute([1, 0, 2])
        output, (hidden, cell) = self.lstm(x)
        # use h from all three layers to calculate mu and sigma
        # (num_layers, batch_size, hidden_dim)
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)  # softplus to make sure standard deviation is positive
        return torch.squeeze(mu), torch.squeeze(sigma)

    def loss_function(self, batch):
        x, y, flag = batch
        # mu: [batch_size, output]
        mu, sigma = self.forward(x)
        loss = 0.
        # sigma: [batch_size, output]
        preds = list()
        for i in torch.arange(self.output_size):
            distribution = torch.distributions.normal.Normal(mu[:, i], sigma[:, i])
            likelihood = distribution.log_prob(y[:, i]) * flag[:, i]
            loss -= torch.mean(likelihood)
            pred = distribution.sample()
            preds.append(pred)
        # [batch_size, output_size]
        preds = torch.stack(preds, dim=-1)
        return loss, preds

class DeepAR_exp(Pytorch_DNN_exp):
    def __init__(self, file, param_dict, config):
        super().__init__(file, param_dict, config)

        self.dataloader = HistoryConcatLoader(
            file,
            param_dict,
            config,
        )
        self.model = self.load_model()
        print_network(self.model)

    def load_model(self):
        model = DeepAR_model(
            sliding_window=self.param_dict['sliding_window'],
            external_flag=self.param_dict['external_flag'],
            external_features=self.config['exp_params']['external_features'],
            history_column_names=self.dataloader.history_column_names,
            target_val_column_names=self.dataloader.target_val_column_names,
            dropout=self.param_dict['dropout'],
            normalization=self.param_dict['normalization'],
            hidden_dim=self.param_dict['hidden_dim'],
            hidden_layers=self.param_dict['hidden_layers']
        )
        return model

    def test_step(self, batch, batch_idx):
        ID, x = batch
        mu, sigma = self.model.forward(x)
        prediction = list()
        for i in torch.arange(self.model.output_size):
            distribution = torch.distributions.normal.Normal(mu[:, i], sigma[:, i])
            pred = distribution.sample()
            prediction.append(pred)
        # [batch_size, output_size]
        prediction = torch.stack(prediction, dim=-1)
        return ID, mu, sigma, prediction

    def test_epoch_end(self, outputs):
        interval_multiplier = prediction_interval_multiplier[str(self.config['exp_params']['prediction_interval'])]
        IDs, mus, sigmas, predictions_mean = list(), list(), list(), list()
        for output in outputs:
            IDs += list(output[0].cpu().detach().numpy())
            mus.append(output[1].cpu().detach().numpy())
            sigmas.append(output[2].cpu().detach().numpy())
            predictions_mean.append(output[3].cpu().detach().numpy())
        IDs, predictions_mean = np.array(IDs), np.concatenate(predictions_mean, axis=0)
        mus, sigmas = np.concatenate(mus, axis=0), np.concatenate(sigmas, axis=0)
        predictions_L = mus - sigmas * interval_multiplier
        predictions_U = mus + sigmas * interval_multiplier

        if self.param_dict['normalization'] != 'none':
            predictions_mean = self.dataloader.scalar_y.inverse_transform(predictions_mean)
            predictions_L = self.dataloader.scalar_y.inverse_transform(predictions_L)
            predictions_U = self.dataloader.scalar_y.inverse_transform(predictions_U)

        cur_logger_folder = f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"

        predictions_frame = pd.DataFrame(
            data=np.stack([predictions_mean, predictions_U, predictions_L], axis=-1).reshape([len(IDs), len(self.dataloader.target_val_column_names)*3]),
            columns=[i.replace('val', 'mean') for i in self.dataloader.target_val_column_names] +
                    [i.replace('val', 'U') for i in self.dataloader.target_val_column_names] +
                    [i.replace('val', 'L') for i in self.dataloader.target_val_column_names]
        )

        predictions_frame['ID'] = IDs
        predictions_frame.to_csv(os.path.join(cur_logger_folder, self.dataloader.file.split('/')[-1]), index=False, columns=['ID']+list(predictions_frame))

        return


def grid_search_DeepAR(config, num_files):
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

        'hidden_dim': config['model_params']['hidden_dim'],
        'hidden_layers': config['model_params']['hidden_layers'],
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
            for one_variate in config['exp_params']['variate']:
                summary = OrderedDict()
                if one_variate == 'multi':
                    cur_param_dict_list = deepcopy(param_dict_list)
                    for one_set in cur_param_dict_list:
                        one_set['variate'] = 'multi'
                    Pytorch_DNN_validation(os.path.join(data_folder, file), cur_param_dict_list, cur_log_dir, config, DeepAR_exp)
                    """
                    hyperparameters selection
                    """
                    for param_index, param_dict in enumerate(cur_param_dict_list):
                        param_dict = OrderedDict(param_dict)
                        setting_name = 'param'
                        for key, val in param_dict.items():
                            setting_name += f'_{key[0].capitalize()}{val}'

                        model_list = [i for i in os.listdir(os.path.join(cur_log_dir, setting_name, 'version_0')) if i.endswith('.ckpt')]
                        assert len(model_list) == 1
                        perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('.ckpt')])
                        # perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('-std')])
                        # std = float(model_list[0][model_list[0].find('-std=')+len('-std='):model_list[0].find('.ckpt')])
                        summary['_'.join(map(str, list(param_dict.values())))] = perf
                elif one_variate == 'uni':
                    cur_param_dict_list = deepcopy(param_dict_list)
                    for one_set in cur_param_dict_list:
                        one_set['variate'] = 'uni'
                    parent_log_dir = os.path.join(cur_log_dir, 'uni')
                    for task_name in task_prediction_horizon.keys():
                        cur_sub_log_dir = os.path.join(parent_log_dir, task_name)
                        if not os.path.exists(cur_sub_log_dir):
                            os.makedirs(cur_sub_log_dir)
                        cur_config = deepcopy(config)
                        cur_config['task_name'] = task_name
                        Pytorch_DNN_validation(os.path.join(data_folder, file), cur_param_dict_list, cur_sub_log_dir, cur_config, DeepAR_exp)
                        """
                        hyperparameters selection
                        """
                        for param_index, param_dict in enumerate(cur_param_dict_list):
                            param_dict = OrderedDict(param_dict)
                            setting_name = 'param'
                            for key, val in param_dict.items():
                                setting_name += f'_{key[0].capitalize()}{val}'

                            model_list = [i for i in os.listdir(os.path.join(cur_sub_log_dir, setting_name, 'version_0')) if i.endswith('.ckpt')]
                            assert len(model_list) == 1
                            perf = float(model_list[0][model_list[0].find('avg_val_metric=')+len('avg_val_metric='):model_list[0].find('.ckpt')])
                            if '_'.join(map(str, list(param_dict.values()))) not in summary:
                                summary['_'.join(map(str, list(param_dict.values())))] = [perf]
                            else:
                                summary['_'.join(map(str, list(param_dict.values())))].append(perf)
            with open(os.path.join(cur_log_dir, 'val_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)

            selected_index = np.argmin(np.array(list(summary.values())))
            selected_params = list(summary.keys())[selected_index]
            param_dict = {
                'batch_size': int(selected_params.split('_')[0]),
                'dropout': float(selected_params.split('_')[1]),
                'external_flag': eval(selected_params.split('_')[2]),
                'hidden_dim': int(selected_params.split('_')[3]),
                'hidden_layers': int(selected_params.split('_')[4]),
                'learning_rate': float(selected_params.split('_')[5]),
                'normalization': selected_params.split('_')[6],
                'sliding_window': int(selected_params.split('_')[7]),
                'variate': selected_params.split('_')[8],
            }
            # save param
            with open(os.path.join(cur_log_dir, 'param.json'), 'w') as f:
                json.dump(param_dict, f, indent=4)

        """
        prediction on testing
        """
        with open(os.path.join(cur_log_dir, 'param.json'), 'r') as f:
            param_dict = json.load(f)
        Pytorch_DNN_testing(os.path.join(data_folder, file), param_dict, cur_log_dir, config, DeepAR_exp)

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
    parser.add_argument('--learning_rate', '-learning_rate', type=str, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str)#, default='[1]')


    parser.add_argument('--location_type', '-location_type', type=str, help='list of location_type options')
    parser.add_argument('--external_flag', '-external_flag', type=str, help='list of external_flag options')
    parser.add_argument('--dropout', '-dropout', type=str, help='list of dropout options')
    parser.add_argument('--hidden_dim', '-hidden_dim', type=str, help='list of hidden_dim options')
    parser.add_argument('--hidden_layers', '-hidden_layers', type=str, help='list of hidden_layers options')

    args = vars(parser.parse_args())
    with open('./../configs/DeepAR.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        config['trainer_params']['gpus'] = 0

    grid_search_DeepAR(config, num_files=args['num_files'])



