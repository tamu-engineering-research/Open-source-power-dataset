# Created by xunannancy at 2021/9/25
import warnings
warnings.filterwarnings('ignore')
from BenchmarkModel.EventClassification.evaluating import run_evaluate_classification
from BenchmarkModel.LoadForecasting.evaluating import run_evaluate_forecasting
from BenchmarkModel.SyntheticDataGeneration.evaluating import run_evaluate_generation

import pickle
import pandas as pd

class TimeSeriesEvaluator:
    def __init__(self, task, root='./../PSML/'):
        self.task = task
        self.root = root
        assert self.task in ['classification', 'forecasting', 'generation']

    def eval(self, input_dict):
        if self.task == 'classification':
            result_dict = run_evaluate_classification(root=self.root, input_dict=input_dict)
        elif self.task == 'forecasting':
            result_dict = run_evaluate_forecasting(root=self.root, input_dict=input_dict)
        elif self.task == 'generation':
            result_dict = run_evaluate_generation(root=self.root, input_dict=input_dict)
        return result_dict

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.task)
        if self.task == 'classification':
            desc += '{\'classification\': classification, \'localization\': localization, \'detection\': detection}\n'
            desc += '- classification: numpy ndarray of shape (#samples,)\n'
            desc += '- localization: numpy ndarray of shape (#samples,)\n'
            desc += '- detection: numpy ndarray of shape (#samples,)\n'
            desc += 'where classification stores the predicted fault type,\n'
            desc += 'localization stores the predicted fault location,\n'
            desc += 'detection stores the predicted fault occurrence time.\n'
        elif self.task == 'forecasting':
            desc += '{$1st loc_year$: {\'ID\': ID, \'yl_t+60(mean)\': yl_t+60(mean), : \'yl_t+60(U)\': yl_t+60(U)},  \'yl_t+60(L)\':  yl_t+60(L), ...}\n$2nd loc_year$: {...}, ...}\n'
            desc += '- $1st loc_year$: file name, e.g., \'CAISO_zone_1_2018\'\n'
            desc += '- ID: testing timestamp ID, numpy ndarray of shape (#samples,)\n'
            desc += '- yl_t+60(mean): t+60 load prediction for PF, numpy ndarray of shape (#samples,)\n'
            desc += '- yl_t+60(U): t+60 load upper prediction for PI, numpy ndarray of shape (#samples,)\n'
            desc += '- yl_t+60(L): t+60 load lower prediction for PI, numpy ndarray of shape (#samples,)\n'

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.task)
        if self.task == 'classification':
            desc += '{\'#samples\': #samples, \'classification\': balanced acc, \'localization\': balanced acc, \'detection\': Macro MAE}\n'
        elif self.task == 'forecasting':
            desc += '{2018: {\'yl_t+60\': {\'#locs:\' #locs, \'RMSE\': RMSE, \'MAE\': MAE, \'MAPE\': MAPE, \'MSIS\': MSIS},\n\'yl_t+1440\':{...}, ...}\n'
            desc += '{2019: {\'yl_t+60\': {\'#locs:\' #locs, \'RMSE\': RMSE, \'MAE\': MAE, \'MAPE\': MAPE, \'MSIS\': MSIS},\n\'yl_t+1440\':{...}, ...}\n'
            desc += '{2020: {\'yl_t+60\': {\'#locs:\' #locs, \'RMSE\': RMSE, \'MAE\': MAE, \'MAPE\': MAPE, \'MSIS\': MSIS},\n\'yl_t+1440\':{...}, ...}\n'

        return desc

def _test_classification_evaluator():
    evaluator = TimeSeriesEvaluator(task='classification', root='/meladyfs/newyork/nanx/freetime/freetime/PowerSystem')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    predictions = pickle.load(open('/meladyfs/newyork/nanx/freetime/examples/power_classification/logs/cnn/version_49/predictions.pkl', 'rb'))
    input_dict = {
        'classification': predictions[:, 0],
        'localization': predictions[:, 1],
        'detection': predictions[:, 2],
    }
    result_dict = evaluator.eval(input_dict)
    return result_dict

def _test_forecasting_evaluator():
    evaluator = TimeSeriesEvaluator(task='forecasting', root='/meladyfs/newyork/nanx/Datasets/PSML')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    predictions = pd.read_csv('/meladyfs/newyork/nanx/freetime/examples/power_forecasting/logs/exponential_smoothing/version_2/CAISO_zone_1_2018.csv')
    input_dict = {
        'CAISO_zone_1_2018': dict(),
    }
    for col_name in list(predictions):
        input_dict['CAISO_zone_1_2018'][col_name] = predictions[col_name].to_numpy()
    result_dict = evaluator.eval(input_dict)
    return result_dict

def _test_generation_evaluator():
    evaluator = TimeSeriesEvaluator(task='generation', root='/meladyfs/newyork/nanx/Datasets/PSML')
    result_dict = evaluator.eval(input_dict = {})
    return result_dict


if __name__ == '__main__':
    # _test_classification_evaluator()
    #_test_forecasting_evaluator()
    _test_generation_evaluator()
    print()

