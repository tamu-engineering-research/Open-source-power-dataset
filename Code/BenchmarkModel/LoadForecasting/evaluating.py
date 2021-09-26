# Created by xunannancy at 2021/9/25
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

task_prediction_horizon = OrderedDict({
    'load': [60, 1440],
    'wind': [5, 30],
    'solar': [5, 30],
})

def perform_evaluate(gt_file, pred):
    gt_data = pd.read_csv(gt_file)
    train_flag = gt_data['train_flag'].to_numpy()
    testing_index = sorted(np.argwhere(train_flag == 0).reshape([-1]))
    gt_testing_data = gt_data.iloc[testing_index]

    pred_data = pd.DataFrame(
        data=np.transpose(np.array(list(pred.values())), (1, 0)),
        columns=list(pred.keys()),
    )

    # combine
    merged_results = pd.merge(left=gt_testing_data, right=pred_data, how='left', on='ID')

    results = dict()

    for task_name, task_prediction_horizon_list in task_prediction_horizon.items():
        for horizon_index, horizon_val in enumerate(task_prediction_horizon_list):
            cur_gt_val, cur_gt_flag, cur_pred_mean = merged_results[f'y{task_name[0]}_t+{horizon_val}(val)'].to_numpy(), merged_results[f'y{task_name[0]}_t+{horizon_val}(flag)'].to_numpy(), merged_results[f'y{task_name[0]}_t+{horizon_val}(mean)'].to_numpy()
            cur_pred_U, cur_pred_L = merged_results[f'y{task_name[0]}_t+{horizon_val}(U)'].to_numpy(), merged_results[f'y{task_name[0]}_t+{horizon_val}(L)'].to_numpy()
            selected_index = sorted(np.argwhere(cur_gt_flag == 1).reshape([-1]))
            valid_gt = cur_gt_val[selected_index]
            val_pred_mean = cur_pred_mean[selected_index]
            val_pred_U, val_pred_L = cur_pred_U[selected_index], cur_pred_L[selected_index]
            results[f'y{task_name[0]}_t+{horizon_val}'] = [valid_gt, val_pred_mean, val_pred_U, val_pred_L]
    return results

def run_evaluate_forecasting(root, input_dict, prediction_interval=0.95):
    data_folder = os.path.join(root, 'processed_dataset', 'forecasting')

    gt_file_dict = dict()
    for i in os.listdir(data_folder):
        cur_year = int(i.split('.')[0].split('_')[-1])
        if cur_year not in gt_file_dict:
            gt_file_dict[cur_year] = [i]
        else:
            gt_file_dict[cur_year].append(i)

    summary = dict()
    for year, file_list in gt_file_dict.items():
        summary[year] = dict()
        file_counter = 0
        total_results = dict()
        for gt_file in tqdm(file_list):
            if gt_file.split('.')[0] not in input_dict:
                continue
            cur_results = perform_evaluate(os.path.join(data_folder, gt_file), input_dict[gt_file.split('.')[0]])
            for key, val in cur_results.items():
                gt, pred_mean, pred_U, pred_L = val[0], val[1], val[2], val[3]
                if key not in total_results:
                    total_results[key] = [gt, pred_mean, pred_U, pred_L]
                else:
                    total_results[key][0] = np.concatenate([total_results[key][0], gt])
                    total_results[key][1] = np.concatenate([total_results[key][1], pred_mean])
                    total_results[key][2] = np.concatenate([total_results[key][2], pred_U])
                    total_results[key][3] = np.concatenate([total_results[key][3], pred_L])

            file_counter += 1
        for key, val in total_results.items():
            gt, pred_mean, pred_U, pred_L = val[0], val[1], val[2], val[3]
            RMSE = np.sqrt(mean_squared_error(gt, pred_mean))
            MAE = mean_absolute_error(gt, pred_mean)
            MAPE = mean_absolute_percentage_error(gt, pred_mean)
            a = prediction_interval
            term1 = pred_U - pred_L
            term2 = 2./a * (pred_L - gt) * (gt < pred_L)
            term3 = 2./a * (gt - pred_U) * (gt > pred_U)
            MSIS = np.mean(term1 + term2 + term3)# / config['exp_params']['naive_scale'][int(horizon_val.split('+')[1])]
            summary[year][key] = {
                '#locs': file_counter,
                'RMSE': RMSE,
                'MAE': MAE,
                'MAPE': MAPE,
                'MSIS': MSIS
            }
    summary = OrderedDict(sorted(summary.items()))
    print(f'summary: {summary}')
    return summary

