import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import torch

class TSDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
class GenerationDataset:
    def __init__(self, root, train_ratio=0.8):
        forced_path = os.path.join(root, 'Millisecond-level PMU Measurements', 'Natural Oscillation')
        self.data_path = os.path.join(root, 'processed_dataset', 'generation.pkl')
        if not os.path.exists(self.data_path):
            if not os.path.exists(os.path.join(root, 'processed_dataset')):
                os.makedirs(os.path.join(root, 'processed_dataset'))
            self.data = self.processing(forced_path, train_ratio)
        else:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)

    def processing(self, path, train_ratio):
        row_list = sorted([int(i.split('_')[1]) for i in os.listdir(path)])
        row_list = [os.path.join(path, f'row_{i}') for i in row_list]
        feature_names = None
        feature_list, label_list = list(), list()
        fault_name_label = OrderedDict({
            'gen_trip': 0,
            'branch_trip': 1,
            'branch_fault': 2,
            'bus_trip': 3,
            'bus_fault': 4,
        })
        print(f'fault_name_label: {fault_name_label}')
        fault_label_name = dict(zip(fault_name_label.values(), fault_name_label.keys()))
        print(f'fault_label_name: {fault_label_name}')

        bus_index = None

        start_index_name_label = None

        for one_file in tqdm(row_list):
            row_index = int(one_file.split('/')[-1].split('_')[1])

            if row_index == 328:
                continue
            info = pd.read_csv(os.path.join(one_file, 'info.csv'))
            info_val = info.to_numpy()
            bus_1, bus_2, type, starttime = int(info_val[0][1].split()[0]), int(info_val[1][1].split()[0]), info_val[2][1].split()[0], float(info_val[3][1].split()[0])
            trans = pd.read_csv(os.path.join(one_file, 'trans.csv'))

            if feature_names is None:
                feature_names = [' '.join(i.split()) for i in trans.columns.to_list()]
            trans.columns = [' '.join(i.split()) for i in trans.columns]
            feature_list.append(trans[feature_names].to_numpy())
            if bus_index is None:
                bus_index = sorted([int(i.split()[1]) for i in feature_names if 'VOLT' in i])
                location_name_label = OrderedDict()
                location_label_counter = 0
                for idx_1, bus1_index in enumerate(bus_index):
                    location_name_label[f'{bus1_index}_-1'] = location_label_counter
                    location_label_counter += 1
                    for idx_2 in range(idx_1+1, len(bus_index)):
                        bus2_index = bus_index[idx_2]
                        location_name_label[f'{bus1_index}_{bus2_index}'] = location_label_counter
                        location_label_counter += 1
                print(f'location_name_label: {len(location_name_label)}, {location_name_label}')
                location_label_name = dict(zip(location_name_label.values(), location_name_label.keys()))
                print(f'location_label_name: {len(location_label_name)}, {location_label_name}')

            if start_index_name_label is None:
                time_list = trans['Time(s)'].to_numpy()
                start_index_name_label = dict(zip(time_list, np.arange(len(time_list))))
                start_index_label_name = dict(zip(np.arange(len(time_list)), time_list))
            if f'{bus_1}_{bus_2}' in location_name_label:
                cur_location = location_name_label[f'{bus_1}_{bus_2}']
            else:
                cur_location = location_name_label[f'{bus_2}_{bus_1}']
            label_list.append([fault_name_label[type], cur_location, start_index_name_label[starttime]])

        print(f'start_index_name_label: {len(start_index_name_label)}, {start_index_name_label}')
        print(f'start_index_label_name: {len(start_index_label_name)}, {start_index_label_name}')
        feature_list = np.array(feature_list)
        label_list = np.array(label_list)

        num_samples = len(feature_list)
        num_train = int(num_samples * train_ratio)
        # permuted_idx = np.random.permutation(np.arange(num_samples))
        permuted_idx = np.arange(num_samples)
        train_idx, test_idx = permuted_idx[:num_train], permuted_idx[num_train:]

        dataset = {
            'feature_names': feature_names,
            'label_names': {
                'fault': fault_name_label,
                'location': location_name_label,
                'starttime': start_index_name_label,
            },

            'feature_list': feature_list,
            'label_list': label_list,

            'data_split': {
                'train': train_idx,
                'test': test_idx
            }
        }

        # print(f'dataset: {dataset}')

        with open(self.data_path, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset

    def load(self, batch_size, shuffle):
        train_idx, test_idx = self.data['data_split']['train'], self.data['data_split']['test']
        train_dataset = TSDataset(
            torch.from_numpy(self.data['feature_list'][train_idx][:,:,1:]).to(torch.float),
            torch.from_numpy(self.data['label_list'][train_idx][:,:1]).to(torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False)
        test_idx = self.data['data_split']['test']
        test_dataset = TSDataset(
            torch.from_numpy(self.data['feature_list'][test_idx][:,:,1:]).to(torch.float),
            torch.from_numpy(self.data['label_list'][train_idx][:,:1]).to(torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

        return train_loader, test_loader
    
