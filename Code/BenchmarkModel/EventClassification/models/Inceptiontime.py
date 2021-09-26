# Created by xunannancy at 2021/9/21
"""
refer from https://github.com/sktime/sktime-dl.git
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_KERAS'] = '1'
from sktime_dl.classification import InceptionTimeClassifier
from tensorflow import keras
import pickle
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
import json
import yaml
from utils import target_name_categories_dict, merge_parameters, fit_prepare, Keras_DNN_exp, Keras_DNN_validation, \
    Keras_DNN_testing, run_evaluate
import argparse
from sklearn.utils import check_random_state
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data
import pandas as pd
import matplotlib.pyplot as plt

class InceptionTimeClassifier_(InceptionTimeClassifier):
    def __init__(
        self,
        nb_filters,
        bottleneck_size,
        depth,
        kernel_size,
        batch_size,
        nb_epochs,
        callbacks,
        random_state,
        verbose,
        model_name,
        model_save_directory,
        learning_rate,
        target_name,
        label_constraints
    ):
        super().__init__(
            nb_filters=nb_filters,
            use_residual=True,
            use_bottleneck=True,
            bottleneck_size=bottleneck_size,
            depth=depth,
            kernel_size=kernel_size,
            batch_size=batch_size,
            nb_epochs=nb_epochs,
            callbacks=callbacks,
            random_state=random_state,
            verbose=verbose,
            model_name=model_name,
            model_save_directory=model_save_directory
        )
        self.learning_rate = learning_rate
        self.target_name = target_name
        self.label_constraints = label_constraints

    def build_model(self, input_shape, nb_classes, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for training

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer
        nb_classes: int
            The number of classes, which shall become the size of the output
             layer

        Returns
        -------
        output : a compiled Keras Model
        """
        input_layer, output_layer = self.build_network(input_shape, **kwargs)

        output_layer = keras.layers.Dense(nb_classes, activation="softmax")(
            output_layer
        )

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

        # if user hasn't provided a custom ReduceLROnPlateau via init already,
        # add the default from literature
        if self.callbacks is None:
            self.callbacks = []

        if not any(
                isinstance(callback, keras.callbacks.ReduceLROnPlateau)
                for callback in self.callbacks
        ):
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=50, min_lr=0.0001
            )
            self.callbacks.append(reduce_lr)

        return model


    def fit(self, X, y, input_checks=True, validation_X=None,
            validation_y=None, **kwargs):
        """
        Fit the classifier on the training set (X, y)

        Parameters
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        y : array-like, shape = [n_instances]
            The training data class labels.
        input_checks : boolean
            whether to check the X and y parameters
        validation_X : a nested pd.Dataframe, or array-like of shape =
        (n_instances, series_length, n_dimensions)
            The validation samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
            Unless strictly defined by the user via callbacks (such as
            EarlyStopping), the presence or state of the validation
            data does not alter training in any way. Predictions at each epoch
            are stored in the model's fit history.
        validation_y : array-like, shape = [n_instances]
            The validation class labels.

        Returns
        -------
        self : object
        """
        self.random_state = check_random_state(self.random_state)
        X = check_and_clean_data(X, y, input_checks=input_checks)

        self.label_encoder, self.onehot_encoder = fit_prepare(y, validation_y, self.target_name, self.label_constraints)
        y_onehot = self.convert_y(y, self.label_encoder, self.onehot_encoder)
        self.classes_ = self.label_encoder.classes_
        self.nb_classes = len(self.classes_)

        validation_data = \
            check_and_clean_validation_data(validation_X, validation_y,
                                            self.label_encoder,
                                            self.onehot_encoder)

        # ignore the number of instances, X.shape[0],
        # just want the shape of each instance
        self.input_shape = X.shape[1:]

        if self.batch_size is None:
            self.batch_size = int(min(X.shape[0] / 10, 16))
        else:
            self.batch_size = self.batch_size

        self.model = self.build_model(self.input_shape, self.nb_classes)

        if self.verbose:
            self.model.summary()

        self.history = self.model.fit(
            X,
            y_onehot,
            batch_size=self.batch_size,
            epochs=self.nb_epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_data=validation_data,
        )
        self._is_fitted = True

        return self

class Inceptiontime_exp(Keras_DNN_exp):
    def __init__(self, log_dir, data_path, param_dict, config):
        super().__init__(log_dir, data_path, param_dict, config)

        self.model = self.load_model()

    def load_model(self):
        checkpoint = super().load_model()

        model = InceptionTimeClassifier_(
            nb_filters=self.param_dict['nb_filters'],
            bottleneck_size=self.param_dict['bottleneck_size'],
            depth=self.param_dict['depth'],
            kernel_size=self.param_dict['kernel_size'],
            batch_size=self.param_dict['batch_size'],
            nb_epochs=self.max_epochs,
            callbacks=[checkpoint],
            random_state=self.config['logging_params']['manual_seed'],
            verbose=True,
            model_name=self.config['model_params']['model_name'],
            model_save_directory=self.log_dir,
            learning_rate=self.param_dict['learning_rate'],
            target_name=self.param_dict['target_name'],
            label_constraints=self.param_dict['label_constraints'],
        )
        return model


def grid_search_Inceptiontime(config):
    np.random.seed(config['logging_params']['manual_seed'])
    tf.random.set_seed(config['logging_params']['manual_seed'])

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
        'normalization': config['exp_params']['normalization'],
        'target_name': config['exp_params']['target_name'],
        'label_constraints': config['exp_params']['label_constraints'],

        'nb_filters': config['model_params']['nb_filters'],
        'bottleneck_size': config['model_params']['bottleneck_size'],
        'depth': config['model_params']['depth'],
        'kernel_size': config['model_params']['kernel_size'],
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
        Keras_DNN_validation(data_path, param_dict_list, log_dir, config, Inceptiontime_exp)

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
            perf_list = [float(i[:i.find('.png')].split('_')[2]) for i in os.listdir(os.path.join(log_dir, setting_name)) if i.endswith('.png')]
            assert len(perf_list) == 1
            summary[target_name]['_'.join(map(str, [j for i, j in param_dict.items() if i!='target_name']))] = perf_list[0]

        reference = np.array(list(summary[target_name].values()))
        if target_name in ['fault', 'location']:
            selected_index = np.argmax(reference)
        elif target_name == 'starttime':
            selected_index = np.argmin(reference)
        selected_params = list(summary[target_name].keys())[selected_index]
        param_dict_res[target_name] = {
            'batch_size': int(selected_params.split('_')[0]),
            'bottleneck_size': int(selected_params.split('_')[1]),
            'depth': int(selected_params.split('_')[2]),
            'kernel_size': int(selected_params.split('_')[3]),
            'label_constraints': eval(selected_params.split('_')[4]),
            'learning_rate': float(selected_params.split('_')[5]),
            'nb_filters': int(selected_params.split('_')[6]),
            'normalization': selected_params.split('_')[7],
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
    Keras_DNN_testing(data_path, param_dict, log_dir, config, Inceptiontime_exp)

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
    parser.add_argument('--batch_size', '-batch_size', type=str, help='list of batch_size')
    parser.add_argument('--max_epochs', '-max_epochs', type=int, help='number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type=int, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str) #, default='[1]')

    parser.add_argument('--label_constraints', '-label_constraints', type=str, help='list of optional label constraints')
    parser.add_argument('--target_name', '-target_name', type=str, help='subtasks to complete')

    parser.add_argument('--nb_filters', '-nb_filters', type=str, help='list of nb_filters options')
    parser.add_argument('--bottleneck_size', '-bottleneck_size', type=str, help='list of bottleneck_size options')
    parser.add_argument('--depth', '-depth', type=str, help='list of depth options')
    parser.add_argument('--kernel_size', '-kernel_size', type=str, help='list of kernel_size options')

    args = vars(parser.parse_args())
    with open('./../configs/Inceptiontime.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    print('gpus: ', config['trainer_params']['gpus'])
    if np.sum(config['trainer_params']['gpus']) < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config['trainer_params']['gpus']))

    grid_search_Inceptiontime(config)

    """
    std:
    summary: {'#samples': 110, 'fault': 0.7411532201813531, 'location': -1, 'starttime': -1}    
    summary: {'#samples': 110, 'fault': -1, 'location': 0.22648083623693377, 'starttime': 36.573170731707314}    
    """