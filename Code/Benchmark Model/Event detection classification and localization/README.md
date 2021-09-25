# Event Detection, Classification and Localization
Here we describe how to reproduce the benchmark results for classification tasks given streaming measurements from sensors.
## Relevant Packages Install
- Create and activate anaconda virtual environment
```angular2html
conda create -n EventDetection python=3.7.10
conda activate EventDetection
```
- Install required packages
```angular2html
pip install -r requirements.txt
```

## Benchmark Results Reproduction
You can find all codes of different models in folder `models` and their respective configuration files in folder 
`configs`. Change the configurations in `***.yaml` from `configs` and run `***.py` from `models` for model training
and evaluation. Or you can directly run `***.py` from `models` with configuration hyperparameters in the command line.

<span style="color:red">**NOTE:**</span> The `data_path` in configuration files is the path to **processed** `millisecond-level PMU Measurements` dataset. For experiment log, 
you can check the folder named `logs`, or you can name your own log folder by setting `logging_params:save_dir` in 
configuration files.

#### Implementation Details
Configurations in `config` folder are consistent of what we've tried in dataset benchmark submission, except the 
`max_epochs` and `manual_seed`. By default, we train all trainable models with 10 random seeds for 50 epochs, and 
the average performance is reported in the end. GPU acceleration is supported for both Pytorch and Tensorflow.

## References
1. **InceptionTime**:
   
    <em>Fawaz, Hassan Ismail, et al. "Inceptiontime: Finding alexnet for time series classification." Data Mining and Knowledge Discovery 34.6 (2020): 1936-1962.</em>
    https://github.com/sktime/sktime-dl.git
1. **MC-DCNN**:
   
   <em>Zheng, Yi, et al. "Time series classification using multi-channels deep convolutional neural networks." International conference on web-age information management. Springer, Cham, 2014.</em>
   
    https://github.com/sktime/sktime-dl.git
1. **ResNet**:
   
   <em>Wang, Zhiguang, Weizhong Yan, and Tim Oates. "Time series classification from scratch with deep neural networks: A strong baseline." 2017 International joint conference on neural networks (IJCNN). IEEE, 2017.</em>
   
    https://github.com/sktime/sktime-dl.git
1. **MLSTM-FCN**:
    
    <em>Karim, Fazle, et al. "Multivariate LSTM-FCNs for time series classification." Neural Networks 116 (2019): 237-245.</em>

    https://github.com/houshd/MLSTM-FCN.git
1. **TapNet**:
   
   <em>Zhang, Xuchao, et al. "Tapnet: Multivariate time series classification with attentional prototypical network." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 04. 2020.</em>
   
   https://github.com/xuczhang/tapnet.git
1. **MiniRocket**:
   
    <em>Dempster, Angus, Daniel F. Schmidt, and Geoffrey I. Webb. "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.</em>
   
    https://github.com/angus924/minirocket.git

