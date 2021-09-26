# Created by xunannancy at 2021/9/25
import os
import pickle
from sklearn.metrics import balanced_accuracy_score
from .models.utils import compute_MMAE

def run_evaluate_classification(root, input_dict):
    data_path = os.path.join(root, 'processed_dataset', 'classification.pkl')

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)

    label_list, data_split = dataset['label_list'], dataset['data_split']
    gt = label_list[data_split['test']]

    # fault type
    classification_acc = balanced_accuracy_score(gt[:, 0], input_dict['classification'])
    # location
    localization_acc = balanced_accuracy_score(gt[:, 1], input_dict['localization'])
    # starttime
    detection_MMAE = compute_MMAE(gt[:, 2], input_dict['detection'])

    summary = {
        '#samples': len(gt),
        'classification': classification_acc,
        'localization': localization_acc,
        'detection': detection_MMAE
    }

    print(f'summary: {summary}')

    return

