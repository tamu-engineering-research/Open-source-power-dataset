# Created by xunannancy at 2021/9/21
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_KERAS'] = '1'
import tensorflow as tf
from utils import Keras_DNN_exp, Keras_DNN_validation, Keras_DNN_testing, merge_parameters, fit_prepare, run_evaluate
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras
from sktime_dl.classification._classifier import BaseDeepClassifier
from sklearn.utils import check_random_state
from sktime_dl.utils import check_and_clean_data, \
    check_and_clean_validation_data
from collections import OrderedDict
import yaml
import json
import numpy as np
from copy import deepcopy
from sklearn.model_selection import ParameterGrid
import argparse

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    # filters = input._keras_shape[-1] # channel_axis = -1 for TF
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


class MLSTM_FCNClassifier(BaseDeepClassifier):
    def __init__(self,
                 batch_size,
                 nb_epochs,
                 callbacks,
                 random_state,
                 verbose,
                 model_name,
                 model_save_directory,
                 learning_rate,
                 target_name,
                 label_constraints):
        super().__init__(model_name=model_name, model_save_directory=model_save_directory)

        self.verbose = verbose
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        self.callbacks = callbacks
        self.random_state = random_state
        self.verbose = verbose
        self._is_fitted = False

        self.learning_rate = learning_rate
        self.target_name = target_name
        self.label_constraints = label_constraints


    def build_model(self, input_shape, nb_classes):
        ip = Input(shape=input_shape) #origin: [None, features, seqlen]

        x = Permute((2, 1))(ip)
        x = Masking()(x)
        x = LSTM(8)(x)
        x = Dropout(0.8)(x)

        # y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = squeeze_excite_block(y)

        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = squeeze_excite_block(y)

        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)

        y = GlobalAveragePooling1D()(y)

        x = concatenate([x, y])

        out = Dense(nb_classes, activation='softmax')(x)

        model = Model(ip, out)

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=["accuracy"],
        )

        if self.callbacks is None:
            self.callbacks = []

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

class MLSTM_FCN_exp(Keras_DNN_exp):
    def __init__(self, log_dir, data_path, param_dict, config):
        super().__init__(log_dir, data_path, param_dict, config)

        self.model = self.load_model()

    def load_model(self):
        checkpoint = super().load_model()

        model = MLSTM_FCNClassifier(
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

def grid_search_MLSTM_FCN(config):
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
        Keras_DNN_validation(data_path, param_dict_list, log_dir, config, MLSTM_FCN_exp)

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
            'label_constraints': eval(selected_params.split('_')[1]),
            'learning_rate': float(selected_params.split('_')[2]),
            'normalization': selected_params.split('_')[3],
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
    Keras_DNN_testing(data_path, param_dict, log_dir, config, MLSTM_FCN_exp)

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
    parser.add_argument('--batch_size', '-batch_size', type=str, help='list of batch_size')
    parser.add_argument('--max_epochs', '-max_epochs', type=int, help='number of epochs')
    parser.add_argument('--learning_rate', '-learning_rate', type=int, help='list of learning rate')
    parser.add_argument('--gpus', '-g', type=str)#, default='[1]')

    parser.add_argument('--label_constraints', '-label_constraints', type=str, help='list of optional label constraints')
    parser.add_argument('--target_name', '-target_name', type=str, help='subtasks to complete')

    args = vars(parser.parse_args())
    with open('./../configs/MLSTM_FCN.yaml', 'r') as file:
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

    grid_search_MLSTM_FCN(config)

    """
    none 
    std
    summary: {'#samples': 110, 'fault': 0.679174278407015, 'location': -1, 'starttime': -1}    
    """
