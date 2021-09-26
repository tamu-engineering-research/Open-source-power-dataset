# Created by xunannancy at 2021/9/21
"""
100% sliding window for DTW
two modes:
1) independent
2) dependent
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import yaml
from utils import merge_parameters
import os
import pickle
from tslearn.metrics import dtw_path_from_metric
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from functools import partial
from tqdm import tqdm
import multiprocessing

def DTW_dependent(seq1, seq2, seqlen, feature_dim):
    seq1 = seq1.reshape([seqlen, feature_dim])
    seq2 = seq2.reshape([seqlen, feature_dim])
    _, similarity_score = dtw_path_from_metric(seq1, seq2)
    return similarity_score

def DTW_independent(seq1, seq2, seqlen, feature_dim):
    seq1 = seq1.reshape([seqlen, feature_dim])
    seq2 = seq2.reshape([seqlen, feature_dim])

    similarity_score_list = list()
    for feature_index in range(feature_dim):
        _, similarity_score = dtw_path_from_metric(seq1[:, feature_index], seq2[:, feature_index])
        similarity_score_list.append(similarity_score)
    return np.mean(similarity_score_list)


def run_NNDTW_sklearn(config):
    assert config['model_params']['type'] in ['dependent', 'independent']

    saved_folder = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name']+config['model_params']['type'].upper()[0])
    if config['exp_params']['test_flag']:
        last_version = config['exp_params']['last_version'] - 1
    else:
        if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)
            last_version = -1
        else:
            last_version = sorted([int(i.split('_')[1]) for i in os.listdir(saved_folder)])[-1]
    log_dir = os.path.join(saved_folder, f'version_{last_version+1}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    data_path = config['exp_params']['data_path']

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    feature_list, label_list, data_split = dataset['feature_list'], dataset['label_list'], dataset['data_split']
    train_x, train_y = feature_list[data_split['train']], label_list[data_split['train']]
    test_x = feature_list[data_split['test']]
    seqlen, feature_dim = train_x.shape[1], train_x.shape[2]

    model_path = os.path.join(log_dir, 'model.joblib')
    if not config['exp_params']['test_flag']:
        if config['model_params']['type'] == 'independent':
            neigh = KNeighborsClassifier(n_neighbors=1, metric=partial(DTW_independent, seqlen=seqlen, feature_dim=feature_dim))
        elif config['model_params']['type'] == 'dependent':
            neigh = KNeighborsClassifier(n_neighbors=1, metric=partial(DTW_dependent, seqlen=seqlen, feature_dim=feature_dim))
        neigh.fit(train_x.reshape([-1, seqlen * feature_dim]), train_y)
        print(f'save model at {model_path}')
        joblib.dump(neigh, model_path)

    print(f'load model from {model_path}')
    model = joblib.load(model_path)
    predictions = model.predict(test_x.reshape([-1, seqlen * feature_dim]))

    with open(os.path.join(log_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)

    return

def run_NNDTW(config, num_threads=10):
    assert config['model_params']['type'] in ['dependent', 'independent']

    saved_folder = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name']+config['model_params']['type'].upper()[0])
    if config['exp_params']['test_flag']:
        last_version = config['exp_params']['last_version'] - 1
    else:
        if not os.path.exists(saved_folder):
            os.makedirs(saved_folder)
            last_version = -1
        else:
            last_version = sorted([int(i.split('_')[1]) for i in os.listdir(saved_folder)])[-1]

    log_dir = os.path.join(saved_folder, f'version_{last_version+1}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    data_path = config['exp_params']['data_path']

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    feature_list, label_list, data_split = dataset['feature_list'], dataset['label_list'], dataset['data_split']
    train_x, train_y = feature_list[data_split['train']], label_list[data_split['train']]
    test_x = feature_list[data_split['test']]
    seqlen, feature_dim = train_x.shape[1], train_x.shape[2]

    if config['model_params']['type'] == 'independent':
        dist_func = partial(DTW_independent, seqlen=seqlen, feature_dim=feature_dim)
    elif config['model_params']['type'] == 'dependent':
        dist_func = partial(DTW_dependent, seqlen=seqlen, feature_dim=feature_dim)
    existing_test_indices = sorted([int(i[i.find('_')+1:i.find('.pkl')]) for i in os.listdir(log_dir)])
    remaining_test_indices = sorted(list(set(range(len(test_x))) - set(existing_test_indices)))
    iterations = int(np.ceil(len(remaining_test_indices)/num_threads))
    for iteration_index in range(iterations):
        print(f'iteration_index: {iteration_index}/{iterations}')
        param_list = list()
        for i in range(iteration_index*num_threads, min((iteration_index+1)*num_threads, len(remaining_test_indices))):
            param_list.append([data_path, remaining_test_indices[i], log_dir, dist_func])
        pool = multiprocessing.Pool()
        pool.map(_nearest_neighbor_searching, param_list)
        pool.close()
        pool.join()

    # collect all samples:
    predictions = list()
    for test_idx in range(len(test_x)):
        predictions.append(pickle.load(open(os.path.join(log_dir, f'predictions_{test_idx}.pkl'), 'rb')))
        # os.remove(os.path.join(log_dir, f'predictions_{test_idx}.pkl'))

    predictions = np.array(predictions)

    with open(os.path.join(log_dir, f'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)

    if not os.path.exists(os.path.join(log_dir, 'config.yaml')):
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
    return

def _nearest_neighbor_searching(param):
    data_path, test_idx, log_dir, dist_func = param

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    feature_list, label_list, data_split = dataset['feature_list'], dataset['label_list'], dataset['data_split']
    train_x, train_y = feature_list[data_split['train']], label_list[data_split['train']]
    test_x = feature_list[data_split['test']]

    distance_list = list()
    for train_idx in tqdm(range(len(train_x))):
        distance = dist_func(test_x[test_idx], train_x[train_idx])
        distance_list.append(distance)
    # save
    with open(os.path.join(log_dir, f'predictions_{test_idx}.pkl'), 'wb') as f:
        pickle.dump(train_y[np.argmin(distance_list)], f)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--type', '-type', type=str, help='either independent or dependent for all feature dimensions')

    args = vars(parser.parse_args())
    with open('./../configs/NNDTW.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    # run_NNDTW(config)
    run_NNDTW(config, num_threads=10)

