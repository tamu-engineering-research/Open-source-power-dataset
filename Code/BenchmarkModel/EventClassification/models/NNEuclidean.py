# Created by xunannancy at 2021/9/21
"""
1-NN Euclidean distance
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import yaml
from utils import merge_parameters, num_features, seqlen, run_evaluate
import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def run_NNEuclidean(config):
    saved_folder = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
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

    if config['exp_params']['normalization'] != 'none':
        if config['exp_params']['normalization'] == 'minmax':
            scalar_x = MinMaxScaler()
        elif config['exp_params']['normalization'] == 'standard':
            scalar_x = StandardScaler()
        scalar_x = scalar_x.fit(train_x.reshape([len(train_y)*seqlen, num_features]))
        train_x = scalar_x.transform(train_x.reshape([len(train_y)*seqlen, num_features])).reshape([len(train_y), seqlen, num_features])
        test_size = len(test_x)
        test_x = scalar_x.transform(test_x.reshape([test_size * seqlen, num_features])).reshape([test_size, seqlen, num_features])

    model_path = os.path.join(log_dir, 'model.joblib')
    if not config['exp_params']['test_flag']:
        neigh = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
        neigh.fit(train_x.reshape([-1, seqlen * feature_dim]), train_y)
        print(f'save model at {model_path}')
        joblib.dump(neigh, model_path)


    print(f'load model from {model_path}')
    model = joblib.load(model_path)
    predictions = model.predict(test_x.reshape([-1, seqlen * feature_dim]))

    with open(os.path.join(log_dir, 'predictions.pkl'), 'wb') as f:
        pickle.dump(predictions, f)

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

    args = vars(parser.parse_args())
    with open('./../configs/NNEuclidean.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config = merge_parameters(args, config)
    print(f'after merge: config, {config}')

    run_NNEuclidean(config)

    """
    minmax:
    summary: {'#samples': 110, 'fault': 0.6748065233998738, 'location': 0.5150019357336431, 'starttime': 29.152439024390244}
    std:
    summary: {'#samples': 110, 'fault': 0.70933669910652, 'location': 0.5113240418118468, 'starttime': 33.32723577235772}
    
    """
