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

## Power Classification
You can find all codes of different models in folder `models` and their respective configuration files in folder 
`configs`. Change the configurations in `***.yaml` from `configs` and run `***.py` from `models` for model training
and evaluation. Or you can directly run `***.py` from `models` with configuration hyperparameters in the command line.

<span style="color:red">**NOTE:**</span> The `data_path` in configuration files is the path to processed power dataset. For experiment log, 
you can check the folder named `logs`, or you can name your own log folder by setting `logging_params:save_dir` in 
configuration files.

#### Implementation Details
Configurations in `config` folder are consistent of what we've tried in dataset benchmark submission, except the 
`max_epochs` and `manual_seed`. By default, we train all trainable models with 10 random seeds for 50 epochs, and 
the average performance is reported in the end. GPU acceleration is supported for both Pytorch and Tensorflow.

