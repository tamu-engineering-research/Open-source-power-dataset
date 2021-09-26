## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from timegan import timegan
from data_loading import real_data_loading, sine_data_generation
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
import time

## Newtork parameters
parameters = dict()
parameters['module'] = 'gru' 
parameters['hidden_dim'] = 256
parameters['num_layer'] = 2
parameters['iterations'] = 5000
parameters['batch_size'] = 64

# Run TimeGAN
data_npz = np.load("../data/real_train.npz")
ori_data = data_npz["trans"]

start_time = time.time()
generated_data = timegan(ori_data, parameters)   
end_time = time.time()
print('Finish Synthetic Data Generation')

print(end_time-start_time)
print(generated_data.shape)
np.savez('generated_data_3.npz', data_feature=generated_data, time=end_time-start_time)
