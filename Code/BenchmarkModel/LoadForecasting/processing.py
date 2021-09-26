# Created by xunannancy at 2021/9/25
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader

external_feature_names = ['DHI', 'DNI', 'GHI', 'Dew Point', 'Solar Zenith Angle', 'Wind Speed', 'Relative Humidity', 'Temperature']
target_features = ['load_power', 'wind_power', 'solar_power']
holidays = USFederalHolidayCalendar().holidays()
task_prediction_horizon = OrderedDict({
    'load': [60, 1440],
    'wind': [5, 30],
    'solar': [5, 30],
})
step_size = {
    'wind': 1, # predict at every minute
    'solar': 1, # predict at every minute
    'load': 60 # predict at every hour
}

class HistoryConcatTrainDataset(Dataset):
    def __init__(self, x, y, flag):
        """
        for training & validation dataset
        :param x: historical y and external features
        :param y: future y
        :param flag: whether future y is to predict or not; for loss computation
        """
        self.x = x
        self.y = y
        self.flag = flag

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.flag[idx]

class HistoryConcatTestDataset(Dataset):
    def __init__(self, ID, x):
        """
        for testing
        :param ID: testing ID
        :param x: historical features and y
        """
        self.ID = ID
        self.x = x
    def __len__(self):
        return len(self.ID)
    def __getitem__(self, idx):
        return self.ID[idx], self.x[idx]

class ForecastingDataset:
    def __init__(self, root):
        self.raw_data_folder = os.path.join(root, 'Minute-level Load and Renewable')
        self.data_folder = os.path.join(root, 'processed_dataset', 'forecasting')
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
        for loc_index, file in enumerate(sorted(os.listdir(self.raw_data_folder))):
            print(f'{loc_index}/{len(os.listdir(self.raw_data_folder))}')
            if not file.endswith('_.csv'):
                continue
            if not os.path.exists(os.path.join(self.data_folder, file.split('.')[0]+'2018.csv')):
                self.processing(file)
            break

    def processing(self, file):
        data = pd.read_csv(os.path.join(self.raw_data_folder, file))
        time = data['time'].to_numpy()
        time_h = [i.split()[0] for i in time]
        date_pd, date_h = pd.DatetimeIndex(time), pd.DatetimeIndex(time_h)
        month_day = (date_pd.month.astype(float) + date_pd.day.astype(float) / 31).to_numpy().reshape([-1, 1])
        weekday = date_pd.weekday.to_numpy().reshape([-1, 1])
        holiday = (date_h.isin(holidays)).astype(int).reshape([-1, 1])
        date_info = np.concatenate([month_day, weekday, holiday], axis=-1)

        year_2018_index = np.sort(np.argwhere((date_pd >= '2018-01-01 00:00:00') & (date_pd < '2019-01-01 00:00:00')).reshape([-1]))
        year_2019_index = np.sort(np.argwhere((date_pd >= '2019-01-01 00:00:00') & (date_pd < '2020-01-01 00:00:00')).reshape([-1]))
        year_2020_index = np.sort(np.argwhere((date_pd >= '2020-01-01 00:00:00') & (date_pd < '2021-01-01 00:00:00')).reshape([-1]))
        for year in tqdm([2018, 2019, 2020]):
            cur_year_index = locals()[f'year_{year}_index']
            cur_data = data.iloc[cur_year_index]
            cur_date_info = date_info[cur_year_index]
            cur_date_pd = pd.DatetimeIndex(cur_data['time'].to_numpy())
            cur_train_test_split_idx = np.argwhere(cur_date_pd == f'{year}-12-01 00:00:00').reshape([-1])[0]

            cur_processed_data_dict = {
                'task': file.split('.')[0]+str(year),
                'feature_name': ['month_day', 'weekday', 'holiday'],
                'training_data': [cur_date_info[:cur_train_test_split_idx]],
                'testing_data': [cur_date_info[cur_train_test_split_idx:]],
            }
            # external features
            cur_processed_data_dict['feature_name'] += external_feature_names
            cur_processed_data_dict['training_data'].append(cur_data[external_feature_names].to_numpy()[:cur_train_test_split_idx])
            cur_processed_data_dict['testing_data'].append(cur_data[external_feature_names].to_numpy()[cur_train_test_split_idx:])

            #cur target features
            for one_target in target_features:
                cur_processed_data_dict['feature_name'] += [f'y{one_target[0]}_t']
                cur_target_data_training = cur_data[[one_target]].to_numpy()[:cur_train_test_split_idx]
                cur_target_data_testing = cur_data[[one_target]].to_numpy()[cur_train_test_split_idx:]
                cur_processed_data_dict['training_data'].append(cur_target_data_training)
                cur_processed_data_dict['testing_data'].append(cur_target_data_testing)

                # every minute or every hour prediction
                predict_flag_training = (np.arange(cur_train_test_split_idx) % step_size[one_target.split('_')[0]] == 0)
                predict_flag_testing = (np.arange(len(cur_data) - cur_train_test_split_idx) % step_size[one_target.split('_')[0]] == 0)
                for forecast_horizon_index, forecast_horizon_val in enumerate(task_prediction_horizon[one_target.split('_')[0]]):
                    cur_processed_data_dict['feature_name'] += [f'y{one_target[0]}_t+{forecast_horizon_val}(val)', f'y{one_target[0]}_t+{forecast_horizon_val}(flag)']
                    # training
                    cur_target_training = np.concatenate([cur_target_data_training[forecast_horizon_val:], np.repeat([[-1]], forecast_horizon_val, axis=0)], axis=0)
                    horizon_predict_flag = (np.arange(len(cur_target_data_training)) < len(cur_target_data_training) - forecast_horizon_val)
                    time_stamp_flag = (cur_target_data_training.squeeze() > 1e-8) & (cur_target_training.squeeze() > 1e-8)
                    cur_target_predict_flag = predict_flag_training & horizon_predict_flag & time_stamp_flag
                    cur_processed_data_dict['training_data'].append(cur_target_training)
                    cur_processed_data_dict['training_data'].append(np.expand_dims(cur_target_predict_flag, axis=-1))
                    # testing
                    cur_target_testing = np.concatenate([cur_target_data_testing[forecast_horizon_val:], np.repeat([[-1]], forecast_horizon_val, axis=0)], axis=0)
                    horizon_predict_flag = (np.arange(len(cur_target_data_testing)) < len(cur_target_data_testing) - forecast_horizon_val)
                    time_stamp_flag = (cur_target_data_testing.squeeze() > 1e-8) & (cur_target_testing.squeeze() > 1e-8)
                    cur_target_predict_flag = predict_flag_testing & horizon_predict_flag & time_stamp_flag
                    cur_processed_data_dict['testing_data'].append(cur_target_testing)
                    cur_processed_data_dict['testing_data'].append(np.expand_dims(cur_target_predict_flag, axis=-1))
            cur_processed_data_dict['training_data'] = np.concatenate(cur_processed_data_dict['training_data'], axis=-1)
            cur_processed_data_dict['testing_data'] = np.concatenate(cur_processed_data_dict['testing_data'], axis=-1)
            training_frame = pd.DataFrame(cur_processed_data_dict['training_data'], columns=cur_processed_data_dict['feature_name'])
            testing_frame = pd.DataFrame(cur_processed_data_dict['testing_data'], columns=cur_processed_data_dict['feature_name'])
            total_frame = pd.concat([training_frame, testing_frame], ignore_index=True)
            total_frame['train_flag'] = np.concatenate([np.ones(training_frame.shape[0]), np.zeros(testing_frame.shape[0])], axis=0)
            total_frame['ID'] = range(total_frame.shape[0])
            total_frame.to_csv(os.path.join(self.data_folder, file.split('.')[0]+str(year)+'.csv'), index=False, header=True, columns=['ID']+list(total_frame)[:-1])
        return

    def load(self, sliding_window, loc, year, batch_size, shuffle):
        data = pd.read_csv(os.path.join(self.data_folder, f'{loc}_{year}.csv'))
        train_flag = data['train_flag'].to_numpy()
        training_index = np.sort(np.argwhere(train_flag == 1).reshape([-1]))[sliding_window:]

        self.history_column_names = list()
        self.target_val_column_names = list()
        for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
            self.history_column_names.append(f'y{task_name[0]}_t')
            for horizon_val in task_prediction_horizon_list:
                self.target_val_column_names.append(f'y{task_name[0]}_t+{horizon_val}(val)')
        self.target_flag_column_names = [i.replace('val', 'flag') for i in self.target_val_column_names]

        y_t = data[self.history_column_names].to_numpy()
        external_features = data[external_feature_names].to_numpy()

        history_y_t = list()
        for index in range(sliding_window, -1, -1):
            history_y_t.append(y_t[training_index-index])
            history_y_t.append(external_features[training_index-index])
        history_y_t = np.concatenate(history_y_t, axis=-1)

        training_validation_target_val = data[self.target_val_column_names].to_numpy()[training_index[sliding_window:]]
        training_validation_target_flag = data[self.target_flag_column_names].to_numpy()[training_index[sliding_window:]]
        selected_index = np.argwhere(np.prod(training_validation_target_flag, axis=-1) == 1).reshape([-1])

        train_x, train_y = history_y_t[selected_index], training_validation_target_val[selected_index]

        train_dataset = HistoryConcatTrainDataset(
            torch.from_numpy(train_x).to(torch.float),
            torch.from_numpy(train_y).to(torch.float),
            torch.from_numpy(training_validation_target_flag[selected_index]).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False)

        """
        prepare testing datasets
        """
        testing_index = np.sort(np.argwhere(train_flag == 0).reshape([-1]))
        testing_data = data.iloc[testing_index]
        testing_ID = testing_data['ID'].to_numpy()
        history_y_t = list()
        for index in range(sliding_window, -1, -1):
            history_y_t.append(y_t[testing_index-index])
            history_y_t.append(external_features[testing_index-index])
        history_y_t = np.concatenate(history_y_t, axis=-1)

        test_dataset = HistoryConcatTestDataset(
            torch.from_numpy(testing_ID).to(torch.int),
            torch.from_numpy(history_y_t).to(torch.float))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

        return train_loader, test_loader



