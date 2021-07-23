# An Open-Source Multi-scale Power System Dataset for Machine Learning on Time Series
This paper aims to bridge the power system and machine learning communities by introducing an open-source multi-time-scale power system data set and providing baseline machine learning models for tasks of practical significance.

## Navigation
- ***Data Released***
  -  **Dataset_full.csv**: full 1-year-long milli-second-level power system time series
  -  **Dataset_classification.csv**: 20-second-long milli-second-level power system time series for classification tasks
  -  **Dataset_forecasting.csv**: 1-year-long hourly power system dataset for forecasting tasks
  

- ***Data Original***
  - **Load**: all collected real hourly load data from multiple sources
  - **Solar**: all collected real 5-min level solar radiance time series data
  - **Wind**: all collected real 1-min level wind speed time series data

- ***Code***
  - **Benchmark Model**
    - *Event Classification and Localization*: baseline models for event classification and localization
    - *Load and Renewable Forecasting*: baseline models for load and renewable forecasting
  - **Joint Simulation**: python codes for joint steady-state and transient simulation between transmission and distribution systems
  - **Data Processing**: python codes for preprocessing original data and estimating load and renewable power 

## Suggested Citation
- Please cite the following paper when you use this data hub:  
`
``An Open-Source Multi-scale Power System Dataset for Machine Learning on Time Series,'' Neural Information Processing System Dataset and Benchmark Track, 2021.
`\
Available at: [arXiv](https://arxiv.org/abs/XXXXXXXXXXXXXX).

- Alternative supplement dataset in Zenodo:
Available at: [Zenodo](https://zenodo.org/deposit/5130612#).

## Contact
