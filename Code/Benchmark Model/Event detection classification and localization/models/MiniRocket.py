# Created by xunannancy at 2021/9/21
"""
use package tsai and document:
https://github.com/angus924/minirocket.git
"""
import warnings
warnings.filterwarnings('ignore')
from numba import njit, prange, vectorize
import numpy as np
import os
import pickle
from sklearn.linear_model import RidgeClassifierCV
import joblib
import pandas as pd
from utils import target_name_categories_dict, compute_MMAE, merge_parameters
from sklearn.metrics import balanced_accuracy_score
from collections import OrderedDict
import json
import yaml
import argparse

@njit("float32[:](float32[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:])", fastmath = True, parallel = False, cache = True)
def _fit_biases(X, num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, quantiles):

    num_examples, num_channels, input_length = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype = np.float32)

    feature_index_start = 0

    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            num_channels_this_combination = num_channels_per_combination[combination_index]

            num_channels_end = num_channels_start + num_channels_this_combination

            channels_this_combination = channel_indices[num_channels_start:num_channels_end]

            _X = X[np.random.randint(num_examples)][channels_this_combination]

            A = -_X          # A = alpha * X = -X
            G = _X + _X + _X # G = gamma * X = 3X

            C_alpha = np.zeros((num_channels_this_combination, input_length), dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels_this_combination, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis = 0)

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

            combination_index += 1
            num_channels_start = num_channels_end

    return biases

def _fit_dilations(input_length, num_features, max_dilations_per_kernel):

    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = \
    np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32) # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation

# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype = np.float32)

def fit(X, num_features = 10_000, max_dilations_per_kernel = 32):

    _, num_channels, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    num_dilations = len(dilations)
    num_combinations = num_kernels * num_dilations

    max_num_channels = min(num_channels, 9)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent, num_combinations)).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype = np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(num_channels, num_channels_this_combination, replace = False)

        num_channels_start = num_channels_end

    biases = _fit_biases(X, num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, quantiles)

    return num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases

# _PPV(C, b).mean() returns PPV for vector C (convolution output) and scalar b (bias)
@vectorize("float32(float32,float32)", nopython = True, cache = True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0

@njit("float32[:,:](float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])))", fastmath = True, parallel = True, cache = True)
def transform(X, parameters):

    num_examples, num_channels, input_length = X.shape

    num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((num_examples, num_features), dtype = np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((num_channels, input_length), dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[combination_index]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha[channels_this_combination] + \
                    C_gamma[index_0][channels_this_combination] + \
                    C_gamma[index_1][channels_this_combination] + \
                    C_gamma[index_2][channels_this_combination]
                C = np.sum(C, axis = 0)

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()
                else:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], biases[feature_index_start + feature_count]).mean()

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

    return features

def grid_search_MiniRocket(config):
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
            # test log should exist
            assert os.path.exists(log_dir)
            flag = False
        else:
            # train log should not exist
            try:
                os.makedirs(log_dir)
                flag = False
            except:
                flag = True

    print(f'log_dir: {log_dir}')

    data_path = config['exp_params']['data_path']

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    feature_list, label_list, data_split = dataset['feature_list'].astype(np.float32), dataset['label_list'], dataset['data_split']
    train_valid_x, train_valid_y = feature_list[data_split['train']], label_list[data_split['train']]
    test_x = feature_list[data_split['test']]

    num_train = int(len(train_valid_x) * config['exp_params']['train_valid_ratio'])
    new_indices = np.random.permutation(range(len(train_valid_x)))

    # transform
    # [batch, seqlen, feature_dim] => [batch, feature_dim, seqlen]
    parameters = fit(train_valid_x.transpose([0, 2, 1]))
    train_valid_x_transform = transform(train_valid_x.transpose([0, 2, 1]), parameters)
    test_x_transform = transform(test_x.transpose([0, 2, 1]), parameters)

    new_true_label_mapping_dict = dict()

    if not config['exp_params']['test_flag']:
        print('training...')
        for target_name in config['exp_params']['target_name']:
            print(f'target_name: {target_name}')
            if target_name == 'fault':
                target_index = 0
                fault_flag = False
            elif target_name == 'location':
                target_index = 1
            elif target_name == 'starttime':
                target_index = 2
            cur_train_valid_y = train_valid_y[:, target_index]
            for label_constraints in config['exp_params']['label_constraints']:
                if target_name == 'fault' and fault_flag:
                    continue
                if label_constraints and target_name != 'fault':
                    y_occ_dict = pd.DataFrame(cur_train_valid_y).groupby([0]).indices
                    new_true_label_mapping = dict(zip(range(len(y_occ_dict)), y_occ_dict.keys()))
                    true_new_label_mapping = dict(zip(y_occ_dict.keys(), range(len(y_occ_dict))))
                    cur_train_valid_y = np.array(list(map(true_new_label_mapping.__getitem__, cur_train_valid_y)))
                else:
                    new_true_label_mapping = true_new_label_mapping = dict(zip(range(target_name_categories_dict[target_name]), range(target_name_categories_dict[target_name])))

                classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
                classifier.fit(train_valid_x_transform[new_indices[:num_train]], cur_train_valid_y[new_indices[:num_train]])
                valid_pred = classifier.predict(train_valid_x_transform[new_indices[num_train:]])
                if target_name in ['fault', 'location']:
                    # balanced accuracy
                    perf = balanced_accuracy_score(cur_train_valid_y[new_indices[num_train:]], valid_pred)
                elif target_name == 'starttime':
                    perf = compute_MMAE(cur_train_valid_y[new_indices[num_train:]], valid_pred)
                # save model
                joblib.dump(classifier, os.path.join(log_dir, f'{target_name}_{label_constraints}_{perf}.joblib'))
                new_true_label_mapping_dict[f'{target_name}_{label_constraints}_{perf}.joblib'] = new_true_label_mapping
                if target_name == 'fault':
                    fault_flag = True

        """
        hyperparameters selection
        """
        summary = dict()
        for i in os.listdir(log_dir):
            if i.endswith('.joblib'):
                target_name, label_constraints, perf = i[:i.find('.joblib')].split('_')
                if target_name not in summary:
                    summary[target_name] = OrderedDict({i: float(perf)})
                else:
                    summary[target_name][i] = float(perf)
        with open(os.path.join(log_dir, 'val_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        param_dict = dict()
        for target_name, target_val in summary.items():
            if target_name in ['fault', 'location']:
                selected_model = list(target_val.keys())[np.argmax(list(target_val.values()))]
            elif target_name == 'starttime':
                selected_model = list(target_val.keys())[np.argmin(list(target_val.values()))]
            param_dict[target_name] = selected_model
        with open(os.path.join(log_dir, 'param.json'), 'w') as f:
            json.dump(param_dict, f, indent=4)
        with open(os.path.join(log_dir, 'new_true_label_mapping_dict.pkl'), 'wb') as f:
            pickle.dump(new_true_label_mapping_dict, f)

    """
    prediction on testing
    """
    with open(os.path.join(log_dir, 'param.json'), 'r') as f:
        param_dict = json.load(f)

    with open(os.path.join(log_dir, 'new_true_label_mapping_dict.pkl'), 'rb') as f:
        new_true_label_mapping_dict = pickle.load(f)
    predictions = list()
    print('testing...')
    for target_name in config['exp_params']['target_name']:
        print(f'target_name: {target_name}')
        model_path = param_dict[target_name]
        print(f'load model for task {target_name} from {model_path}')
        classifier = joblib.load(os.path.join(log_dir, model_path))
        test_pred = classifier.predict(test_x_transform)
        new_true_label_mapping = new_true_label_mapping_dict[model_path]
        predictions.append(np.array(list(map(new_true_label_mapping.__getitem__, test_pred))))
    predictions = np.stack(predictions, axis=-1)
    with open(os.path.join(log_dir, f'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)

    if not os.path.exists(os.path.join(log_dir, 'config.yaml')):
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--train_valid_ratio', '-train_valid_ratio', type=float, help='select hyperparameters on validation set')
    parser.add_argument('--label_constraints', '-label_constraints', type=str, help='list of optional label constraints')
    parser.add_argument('--target_name', '-target_name', type=str, help='subtasks to complete')
    parser.add_argument('--manual_seed', '-manual_seed', type=int, help='manual_seed')

    args = vars(parser.parse_args())
    with open('./../configs/MiniRocket.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    grid_search_MiniRocket(config)






