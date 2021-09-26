import warnings
warnings.filterwarnings('ignore')
from .BenchmarkModel.EventClassification.processing import ClassificationDataset
from .BenchmarkModel.LoadForecasting.processing import ForecastingDataset

class TimeSeriesLoader():
    def __init__(self, task, root='./../PSML/'):
        """ Initiate data loading for each task
        """
        self.task = task
        if self.task == 'forecasting':
            # Returns load and renewable energy forecasting data
            self.dataset = ForecastingDataset(root)
        elif self.task == 'classification':
            # Returns event detection, classification and localization data
            self.dataset = ClassificationDataset(root)
        elif self.task == 'generation':
            # Returns PMU stream data
            pass
        else:
            raise Exception

    def load(self, batch_size, shuffle, sliding_window=120, loc=None, year=None):
        if self.task == 'forecasting':
            if loc is None:
                loc = 'CAISO_zone_1'
                year = 2018
            train_loader, test_loader = self.dataset.load(sliding_window, loc, year, batch_size, shuffle)
            return train_loader, test_loader
        elif self.task == 'classification':
            train_loader, test_loader = self.dataset.load(batch_size, shuffle)
            return train_loader, test_loader
        elif self.task == 'generation':
            return


def _test_classification_loader():
    loader_ins = TimeSeriesLoader('classification', root='/meladyfs/newyork/nanx/Datasets/PSML')
    train_loader, test_loader = loader_ins.load(batch_size=32, shuffle=True)
    print(f'train_loader: {len(train_loader)}')

    for i in train_loader:
        feature, label = i
        print(f'feature: {feature.shape}')
        print(f'label: {label.shape}')
        break

    print(f'test_loader: {len(test_loader)}')
    for i in test_loader:
        print(f'feature: {i.shape}')

        break
    return

def _test_forecasting_loader():
    loader_ins = TimeSeriesLoader('forecasting', root='/meladyfs/newyork/nanx/Datasets/PSML')
    train_loader, test_loader = loader_ins.load(batch_size=32, shuffle=True)
    print(f'train_loader: {len(train_loader)}')

    for i in train_loader:
        x, y, flag = i
        print(f'x: {x.shape}')
        print(f'y: {y.shape}')
        print(f'flag: {flag.shape}')
        break

    print(f'test_loader: {len(test_loader)}')
    for i in test_loader:
        ID, x = i
        print(f'ID: {ID.shape}')
        print(f'x: {x.shape}')

        break
    return

if __name__ == '__main__':
    # _test_classification_loader()
    # _test_forecasting_loader()
    print()


        