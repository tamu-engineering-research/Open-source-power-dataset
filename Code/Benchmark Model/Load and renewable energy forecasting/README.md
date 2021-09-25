# Load and Renewable Energy Forecasting
Here we describe how to reproduce the benchmark results for point forecast (PF) and 
prediction interval (PI) on load, solar and wind power, given weather and date information.
## Relevant Packages Install
- Create and activate anaconda virtual environment
```angular2html
conda create -n EnergyForecasting python=3.7.10
conda activate EnergyForecasting
```
- Install required packages
```angular2html
pip install -r requirements.txt
```
## Benchmark Results Reproduction
You can find all codes of different models in folder `models` and their respective configuration files in folder 
`configs`. Change the configurations in `***.yaml` from `configs` and run `***.py` from `models` for model training
and evaluation. Or you can directly run `***.py` from `models` with configuration hyperparameters in the command line.

<span style="color:red">**NOTE:**</span> The `data_folder` in configuration files is the path to the **raw** `minute-level load and renewable` dataset folder. For experiment log, 
you can check the folder named `logs`, or you can name your own log folder by setting `logging_params:save_dir` in 
configuration files.

#### Implementation Details
`num_files` in command line indicates the number of location-year for load, solar and wind power forecasting. There are 66 locations
in total and measurements in 2018, 2019 and 2020 are provided. Therefore, setting `num_files=198` will training and testing individual models 
for each location per year.

Configurations in `config` folder are consistent of what we've tried in dataset benchmark submission, except the 
`max_epochs`, which should be 50 by default. GPU acceleration is supported for both Pytorch and Tensorflow.
When considering temporal dependencies, we use `sliding_window=120` by default.
## References

1. **N-BEATS**
   
   <em>Oreshkin, Boris N., et al. "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting." arXiv preprint arXiv:1905.10437 (2019).</em>
   
   https://github.com/ElementAI/N-BEATS.git 
   
1. **WaveNet**
   
   <em>Oord, Aaron van den, et al. "Wavenet: A generative model for raw audio." arXiv preprint arXiv:1609.03499 (2016).</em>
   
   https://github.com/vincentherrmann/pytorch-wavenet.git
   
1. **TCN**
   
   <em>Lea, Colin, et al. "Temporal convolutional networks for action segmentation and detection." proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.</em>
   
   https://github.com/locuslab/TCN.git
   
1. **LSTNet**
      
   <em>Lai, Guokun, et al. "Modeling long-and short-term temporal patterns with deep neural networks." The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 2018.</em>
   
   https://github.com/laiguokun/LSTNet.git
   
1. **DeepAR**
   
   <em>Salinas, David, et al. "DeepAR: Probabilistic forecasting with autoregressive recurrent networks." International Journal of Forecasting 36.3 (2020): 1181-1191.</em>
   
   https://github.com/zhykoties/TimeSeries.git
   
1. **Informer**
   
   <em>Zhou, Haoyi, et al. "Informer: Beyond efficient transformer for long sequence time-series forecasting." Proceedings of AAAI. 2021.</em>
   
   https://github.com/zhouhaoyi/Informer2020.git
   
1. **Neural ODE**
   
   <em>Chen, Ricky TQ, et al. "Neural ordinary differential equations." arXiv preprint arXiv:1806.07366 (2018).</em>
   
   https://github.com/rtqichen/torchdiffeq.git