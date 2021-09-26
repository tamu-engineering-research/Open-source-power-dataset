# Created by xunannancy at 2021/9/21
import pytorch_lightning as pl
import torch
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from torch import optim
from collections import OrderedDict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning import Trainer
from collections import OrderedDict
from copy import deepcopy
from tensorflow import keras
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import yaml

num_features = 92
seqlen = 960

def run_evaluate(config, verbose=True):
    if verbose:
        saved_folder = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
        if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)
            last_version = -1
        else:
            last_version = sorted([int(i.split('_')[1]) for i in os.listdir(saved_folder) if i.startswith('version_')])[-1]
        log_dir = os.path.join(saved_folder, f'version_{last_version+1}')
        os.makedirs(log_dir)
        print(f'log_dir: {log_dir}')

    data_path = config['exp_params']['data_path']

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    label_list, data_split = dataset['label_list'], dataset['data_split']
    gt = label_list[data_split['test']]

    prediction_folder = config['exp_params']['prediction_path']
    predictions = pickle.load(open(os.path.join(prediction_folder, 'predictions.pkl'), 'rb'))
    model_config = yaml.safe_load(open(os.path.join(prediction_folder, 'config.yaml'), 'r'))
    # fault type
    fault_acc = location_acc = starttime_MMAE = -1
    if 'target_name' not in model_config['exp_params']:
        model_config['exp_params']['target_name'] = ['fault', 'location', 'starttime']
    if 'fault' in model_config['exp_params']['target_name']:
        index = model_config['exp_params']['target_name'].index('fault')
        fault_acc = balanced_accuracy_score(gt[:, 0], predictions[:, index])
    # location
    if 'location' in model_config['exp_params']['target_name']:
        index = model_config['exp_params']['target_name'].index('location')
        location_acc = balanced_accuracy_score(gt[:, 1], predictions[:, index])
    # starttime
    if 'starttime' in model_config['exp_params']['target_name']:
        index = model_config['exp_params']['target_name'].index('starttime')
        starttime_MMAE = compute_MMAE(gt[:, 2], predictions[:, index])

    summary = {
        '#samples': len(gt),
        'fault': fault_acc,
        'location': location_acc,
        'starttime': starttime_MMAE
    }
    if verbose:
        with open(os.path.join(log_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    print(f'summary: {summary}')

    return


target_name_categories_dict = OrderedDict({
    'fault': 5,
    'location': 276,
    'starttime': 960
})

target_name_index_dict = {
    'fault': 0,
    'location': 1,
    'starttime': 2
}

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

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %.3f M' % (num_params/1000000))
    return

def compute_MMAE(y, pred):
    """
    macro-mae for start time ordinal regression
    :param y: [batch_size]
    :param pred: [batch_size]
    :return:
    """
    starttime_occ_dict = pd.DataFrame(y).groupby([0]).indices
    starttime_MMAE = list()
    for starttime, indices in starttime_occ_dict.items():
        MAE = np.mean(np.abs(y[indices] - pred[indices]))
        starttime_MMAE.append(MAE)
    starttime_MMAE = np.mean(starttime_MMAE)
    return starttime_MMAE

#<<<<<<<<<<<<<<<<Pytorch models<<<<<<<<<<<<<<<<<<

class TSTrainDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TSTestDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

class TSLoader():
    def __init__(self, data_path, param_dict, config):
        self.data_path = data_path
        self.config = config
        self.param_dict = param_dict
        self.random_seed = config['logging_params']['manual_seed']

        self.train_valid_ratio = self.config['exp_params']['train_valid_ratio']
        self.num_workers = self.config['exp_params']['num_workers']

        self.target_name = self.param_dict['target_name']

        self.batch_size = self.param_dict['batch_size']
        if self.config['model_params']['model_name'] == 'TapNet':
            self.drop_last = True
        else:
            self.drop_last = False

        self.set_dataset()

    def set_dataset(self):
        np.random.seed(self.random_seed)
        with open(self.data_path, 'rb') as f:
            dataset = pickle.load(f)

        feature_list, label_list, data_split = dataset['feature_list'], dataset['label_list'][:, target_name_index_dict[self.target_name]], dataset['data_split']
        train_valid_x, train_valid_y = feature_list[data_split['train']], label_list[data_split['train']]
        # NOTE: reduce label size
        if self.param_dict['label_constraints']:
            y_occ_dict = pd.DataFrame(train_valid_y).groupby([0]).indices
            self.new_true_label_mapping = dict(zip(range(len(y_occ_dict)), y_occ_dict.keys()))
            self.true_new_label_mapping = dict(zip(y_occ_dict.keys(), range(len(y_occ_dict))))
            train_valid_y = np.array(list(map(self.true_new_label_mapping.__getitem__, train_valid_y)))
        else:
            self.new_true_label_mapping = self.true_new_label_mapping = dict(zip(range(target_name_categories_dict[self.target_name]), range(target_name_categories_dict[self.target_name])))
        test_x = feature_list[data_split['test']]

        if 'normalization' in self.param_dict and self.param_dict['normalization'] != 'none':
            if self.param_dict['normalization'] == 'minmax':
                self.scalar_x = MinMaxScaler()
            elif self.param_dict['normalization'] == 'standard':
                self.scalar_x = StandardScaler()
            # [batch_size, seqlen, feature_dim]
            self.scalar_x = self.scalar_x.fit(train_valid_x.reshape([len(data_split['train']) * seqlen, num_features]))

            train_valid_x = self.scalar_x.transform(train_valid_x.reshape([len(data_split['train']) * seqlen, num_features])).reshape([len(data_split['train']), seqlen, num_features])
            test_x = self.scalar_x.transform(test_x.reshape([len(data_split['test']) * seqlen, num_features])).reshape([len(data_split['test']), seqlen, num_features])

        num_train = int(len(train_valid_x) * self.train_valid_ratio)
        new_indices = np.random.permutation(range(len(train_valid_x)))
        train_x, train_y = train_valid_x[new_indices[:num_train]], train_valid_y[new_indices[:num_train]]
        valid_x, valid_y = train_valid_x[new_indices[num_train:]], train_valid_y[new_indices[num_train:]]

        self.train_dataset = TSTrainDataset(
            torch.from_numpy(train_x).to(torch.float),
            torch.from_numpy(train_y).to(torch.long)
        )
        self.valid_dataset = TSTrainDataset(
            torch.from_numpy(valid_x).to(torch.float),
            torch.from_numpy(valid_y).to(torch.long)
        )
        self.test_dataset = TSTestDataset(
            torch.from_numpy(test_x).to(torch.float)
        )
        return

    def load_train(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=self.drop_last)

    def load_valid(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=self.drop_last)

    def load_test(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, drop_last=False)

class Pytorch_DNN_exp(pl.LightningModule):
    def __init__(self, data_path, param_dict, config):
        super().__init__()
        self.save_hyperparameters()

        self.param_dict = param_dict
        self.config = config
        self.max_epochs = config['trainer_params']['max_epochs']
        self.target_name = param_dict['target_name']

    def oracle_loss(self, batch):
        y = batch[1]
        loss, pred = self.model.loss_function(batch)
        if self.target_name in ['fault', 'location']:
            # balanced_acc
            perf = balanced_accuracy_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        elif self.target_name == 'starttime':
            perf = compute_MMAE(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
        return {'loss': loss, 'metric': perf, 'pred': pred, 'y': batch[1]}

    def training_step(self, batch, batch_idx):
        train_loss = self.oracle_loss(batch)
        self.logger.experiment.log({f'train_{key}': float(val) for key, val in train_loss.items() if key not in ['pred', 'y']})
        return train_loss

    def training_epoch_end(self, outputs):
        avg_metric_values = {
            'loss': list(),
        }

        pred_list, y_list = list(), list()
        for output in outputs:
            for key in avg_metric_values.keys():
                avg_metric_values[key].append(output[key])
            pred_list.append(output['pred'])
            y_list.append(output['y'])
        for metric, avg_value in avg_metric_values.items():
            self.log(f'avg_val_{metric}', torch.mean(torch.stack(avg_value)))

        pred_list, y_list = torch.cat(pred_list).cpu().detach().numpy(), torch.cat(y_list).cpu().detach().numpy()
        if self.target_name in ['fault', 'location']:
            # balanced_acc
            perf = balanced_accuracy_score(y_list, pred_list)
        elif self.target_name == 'starttime':
            perf = compute_MMAE(y_list, pred_list)

        self.log(f'avg_val_metric', perf)
        return

    def validation_step(self, batch, batch_idx):
        valid_loss = self.oracle_loss(batch)
        self.logger.experiment.log({f'val_{key}': float(val) for key, val in valid_loss.items() if key not in ['pred', 'y']})
        return valid_loss

    def validation_epoch_end(self, outputs):
        avg_metric_values = {
            'loss': list(),
        }

        pred_list, y_list = list(), list()
        for output in outputs:
            for key in avg_metric_values.keys():
                avg_metric_values[key].append(output[key])
            pred_list.append(output['pred'])
            y_list.append(output['y'])
        for metric, avg_value in avg_metric_values.items():
            self.log(f'avg_train_{metric}', torch.mean(torch.stack(avg_value)))

        pred_list, y_list = torch.cat(pred_list).cpu().detach().numpy(), torch.cat(y_list).cpu().detach().numpy()
        if self.param_dict['target_name'] in ['fault', 'location']:
            # balanced_acc
            perf = balanced_accuracy_score(y_list, pred_list)
        elif self.param_dict['target_name'] == 'starttime':
            perf = compute_MMAE(y_list, pred_list)

        self.log(f'avg_train_metric', perf)

        cur_logger_folder = f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
        if self.current_epoch == self.max_epochs - 1:
            self.plot_write_summmary_train_valid(cur_logger_folder)
        return

    def test_step(self, batch, batch_idx):
        x = batch
        pred = self.model.forward(x)
        return torch.argmax(pred, dim=-1)

    def test_epoch_end(self, outputs):
        predictions = torch.cat(outputs).cpu().detach().numpy()
        cur_logger_folder = f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"

        # map to true labels
        predictions = np.array(list(map(self.dataloader.new_true_label_mapping.__getitem__, predictions)))
        with open(os.path.join(cur_logger_folder, f'predictions_{self.target_name}.pkl'), 'wb') as f:
            pickle.dump(predictions, f)

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
            'metric': 'b'
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

                for y_index, (key, color) in enumerate(key_loss_name.items()):
                    cur_value = df[f'{prefix}{data_type}_{key}'].dropna().to_numpy()[sanity_check:]
                    if y_index == 0:
                        cur_ax = ax
                    else:
                        cur_ax = ax.twinx()
                    cur_ax.plot(range(1, len(cur_value)+1), cur_value, color=color, label=key)
                    cur_ax.set_ylabel(key)

                ax.set_xlabel(f'{value_type.capitalize()}s')
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

def Pytorch_DNN_validation(data_path, param_dict_list, log_dir, config, model_exp):
    for param_index, param_dict in enumerate(param_dict_list):
        param_dict = OrderedDict(param_dict)
        setting_name = param_dict['target_name']
        for key, val in param_dict.items():
            if key == 'target_name':
                continue
            setting_name += f'_{key[0].capitalize()}{val}'
        tt_logger = TestTubeLogger(
            save_dir=log_dir,
            name=setting_name,
            debug=False,
            create_git_tag=False,
            version=0
        )
        if param_dict['target_name'] in ['fault', 'location']:
            cur_mode = 'max'
        elif param_dict['target_name'] == 'starttime':
            cur_mode = 'min'
        checkpoint_callback_V = ModelCheckpoint(
            dirpath=f"{tt_logger.save_dir}/{tt_logger.name}/version_0/",
            filename='best-{epoch:02d}-{avg_val_metric:.3f}',
            save_top_k=1,
            verbose=True,
            mode=cur_mode,
            monitor='avg_val_metric',
        )
        exp = model_exp(
            data_path=data_path,
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

def Pytorch_DNN_testing(data_path, param_dict_list, log_dir, config, model_exp):
    for target_name, param_dict in param_dict_list.items():
        param_dict = OrderedDict(param_dict)
        setting_name = target_name
        for key, val in param_dict.items():
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
        model_path = [i for i in os.listdir(f"{log_dir}/{setting_name}/version_0") if i.endswith('.ckpt') and 'best' in i]
        param_dict['target_name'] = target_name
        exp = model_exp(
            data_path=data_path,
            param_dict=param_dict,
            config=config
        )
        checkpoint_path = os.path.join(f"{log_dir}/{setting_name}/version_0", model_path[0])
        exp = exp.load_from_checkpoint(checkpoint_path)
        print(f'Load checkpoint from {checkpoint_path}...')
        exp.param_dict = param_dict
        runner.test(exp)
    predictions = list()
    for target_name in target_name_categories_dict.keys():
        try:
            cur_file_path = os.path.join(log_dir, f'predictions_{target_name}.pkl')
            with open(cur_file_path, 'rb') as f:
                predictions.append(pickle.load(f))
            os.remove(cur_file_path)
        except FileNotFoundError:
            print(f'cannot find predictions for {target_name}')
    predictions = np.stack(predictions, axis=-1)
    with open(os.path.join(log_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)
    return

#<<<<<<<<<<<<<<<<keras models<<<<<<<<<<<<<<<<<<

class label_encoder:
    def __init__(self, true_new_label_mapping, new_true_label_mapping):
        self.true_new_label_mapping = true_new_label_mapping
        self.new_true_label_mapping = new_true_label_mapping
        self.classes_ = np.array(list(self.true_new_label_mapping.values()))

    def fit_transform(self, y):
        y = np.array(list(map(self.true_new_label_mapping.__getitem__, y)))
        return y

    def inverse_transform(self, y):
        y = list(map(self.new_true_label_mapping.__getitem__, y))
        return y

    def transform(self, y):
        return self.fit_transform(y)

class onehot_encoder:
    def __init__(self, num_labels):
        self.num_labels = num_labels

    def fit_transform(self, y):
        batch_size = y.shape[0]
        y = y.reshape([batch_size])
        new_y = np.zeros([y.shape[0], self.num_labels])
        new_y[np.arange(batch_size), y] = 1
        return new_y

def fit_prepare(y, validation_y, target_name, label_constraints):
    """
    NOTE: validation_y could be none when train and validation includede in y
    :param y:
    :param validation_y:
    :param target_name:
    :param label_constraints:
    :return:
    """
    if label_constraints:
        y_train_valid = np.concatenate([y, validation_y])
        y_occ_dict = pd.DataFrame(y_train_valid).groupby([0]).indices
        new_true_label_mapping = dict(zip(range(len(y_occ_dict)), y_occ_dict.keys()))
        true_new_label_mapping = dict(zip(y_occ_dict.keys(), range(len(y_occ_dict))))
    else:
        new_true_label_mapping = true_new_label_mapping = dict(zip(range(target_name_categories_dict[target_name]), range(target_name_categories_dict[target_name])))

    return label_encoder(true_new_label_mapping, new_true_label_mapping), onehot_encoder(num_labels=len(new_true_label_mapping))

class Keras_DNN_exp():
    def __init__(self, log_dir, data_path, param_dict, config):
        self.data_path = data_path
        self.param_dict = param_dict
        self.config = config
        self.target_name = param_dict['target_name']
        self.max_epochs = config['trainer_params']['max_epochs']

        setting_name = self.param_dict['target_name']
        for key, val in self.param_dict.items():
            if key == 'target_name':
                continue
            setting_name += f'_{key[0].capitalize()}{val}'
        self.log_dir = os.path.join(log_dir, setting_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.load_dataset()

    def load_dataset(self):
        cur_param_dict = deepcopy(self.param_dict)
        cur_param_dict['label_constraints'] = False
        self.dataloader = TSLoader(self.data_path, cur_param_dict, self.config)
        self.train_x = self.dataloader.train_dataset.x.numpy()
        self.train_y = self.dataloader.train_dataset.y.numpy()
        self.valid_x = self.dataloader.valid_dataset.x.numpy()
        self.valid_y = self.dataloader.valid_dataset.y.numpy()
        self.test_x = self.dataloader.test_dataset.x.numpy()
        return

    def load_model(self):
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.log_dir+'/best.hdf5',
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False, #True,
            mode="max"
        )
        return checkpoint

    def train(self):
        self.model.fit(
            X=self.train_x,
            y=self.train_y,
            input_checks=True,
            validation_X=self.valid_x,
            validation_y=self.valid_y
        )
        if self.config['model_params']['model_name'] not in ['MCNN']:
            self.plot_write_summmary_train_valid()
        return

    def test(self):
        model_path = [os.path.join(self.log_dir, i) for i in os.listdir(self.log_dir) if i.endswith('.hdf5')]
        assert len(model_path) == 1
        cur_label_encoder, cur_onehot_encoder = fit_prepare(self.train_y, self.valid_y, self.target_name, self.param_dict['label_constraints'])
        self.model.model = keras.models.load_model(model_path[0])
        self.model.label_encoder, self.model.onehot_encoder = cur_label_encoder, cur_onehot_encoder
        self.model.classes_ = self.model.label_encoder.classes_
        self.model.nb_classes = len(self.model.classes_)

        self.model._is_fitted = True
        if self.config['model_params']['model_name'] == 'MCNN':
            self.model.set_hyperparameters()
            with open(os.path.join(self.log_dir, 'predict_params.json'), 'r') as f:
                predict_params = json.load(f)
            self.model.best_pool_factor = predict_params['best_pool_factor']
            self.model.input_shapes = predict_params['input_shapes']
            self.model.set_hyperparameters()
        predictions = self.model.predict(X=self.test_x)

        with open(os.path.join('/'.join(self.log_dir.split('/')[:-1]), f'predictions_{self.target_name}.pkl'), 'wb') as f:
            pickle.dump(predictions, f)

        return

    def plot_write_summmary_train_valid(self):
        key_loss_name = {
            'loss': 'r',
            'accuracy': 'b'
        }

        train_loss, train_accuracy = np.array(self.model.history.history['loss']), np.array(self.model.history.history['accuracy'])
        val_loss, val_accuracy = np.array(self.model.history.history['val_loss']), np.array(self.model.history.history['val_accuracy'])
        saved_epoch = np.argmax(val_accuracy)

        fig, axes = plt.subplots(1, 2, figsize=(4*2, 3*1))
        for data_index, data_type in enumerate(['train', 'val']):
            for metric_index, metric_type in enumerate(['loss', 'accuracy']):
                if metric_index == 0:
                    ax = axes[data_index]
                    ax.set_xlabel('Epoch')
                else:
                    ax = axes[data_index].twinx()
                cur_value = locals()[f'{data_type}_{metric_type}']
                ax.plot(range(1, len(cur_value)+1), cur_value, color=key_loss_name[metric_type], label=metric_type)
                ax.set_ylabel(metric_type)
                ax.legend(loc=['upper left', 'upper right'][metric_index])
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f'loss_{saved_epoch}_{val_accuracy[saved_epoch]}.png'))
        # plt.show()
        plt.close()

        return

def Keras_DNN_validation(data_path, param_dict_list, log_dir, config, model_exp):
    for param_index, param_dict in enumerate(param_dict_list):
        param_dict = OrderedDict(param_dict)
        exp = model_exp(log_dir, data_path, param_dict, config)
        exp.train()
    return

def Keras_DNN_testing(data_path, param_dict_list, log_dir, config, model_exp):
    for target_name, param_dict in param_dict_list.items():
        param_dict['target_name'] = target_name
        param_dict = OrderedDict(param_dict)
        exp = model_exp(log_dir, data_path, param_dict, config)
        exp.test()
    predictions = list()
    for target_name in target_name_categories_dict.keys():
        try:
            cur_file_path = os.path.join(log_dir, f'predictions_{target_name}.pkl')
            with open(cur_file_path, 'rb') as f:
                predictions.append(pickle.load(f))
            os.remove(cur_file_path)
        except FileNotFoundError:
            print(f'cannot find predictions for {target_name}')
    predictions = np.stack(predictions, axis=-1)
    with open(os.path.join(log_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)
    return

