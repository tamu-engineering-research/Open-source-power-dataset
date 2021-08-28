# PSML: A Multi-scale Time-series Dataset for Machine Learning in Decarbonized Energy Grids
The electric grid is a key enabling infrastructure for the ambitious transition towards carbon neutrality as we grapple with climate change. With deepening penetration of renewable energy resources and electrified transportation, the reliable and secure operation of the electric grid becomes increasingly challenging. In this paper, we present PSML, a first-of-its-kind open-access multi-scale time-series dataset, to aid in the development of data-driven machine learning (ML) based approaches towards reliable operation of future electric grids. The dataset is generated through a novel transmission + distribution (T+D) co-simulation designed to capture the increasingly important interactions and uncertainties of the grid dynamics, containing electric load, renewable generation, weather, voltage and current measurements at multiple spatio-temporal scales. Using PSML, we provide state-of-the-art ML baselines on three challenging use cases of critical importance to achieve: (i) early detection, accurate classification and localization of dynamic disturbance events; (ii) robust hierarchical forecasting of load and renewable energy with the presence of uncertainties and extreme events; and (iii) realistic synthetic generation of physical-law-constrained measurement time series. We envision that this dataset will enable advances for ML in dynamic systems, while simultaneously allowing ML researchers to contribute towards carbon-neutral electricity and mobility. 

## Navigation
- ***Data Released***
  -  **Dataset_full.csv**: full 1-year-long milli-second-level power system time series
  -  **Dataset_classification.csv**: 20-second-long milli-second-level power system time series for classification tasks
  -  **Dataset_forecasting.csv**: 1-year-long hourly power system dataset for forecasting tasks
  

- ***Data Original***
  - **Load**: all collected real hourly load data from multiple sources
  - **Solar**: all collected real 5-min level solar radiance time series data
  - **Wind**: all collected real 1-min level wind speed time series data

- ***Code*** (to be released upon acceptance)
  - **Benchmark Model**
    - *Event Classification and Localization*: baseline models for event classification and localization
    - *Load and Renewable Forecasting*: baseline models for load and renewable forecasting
  - **Joint Simulation**: python codes for joint steady-state and transient simulation between transmission and distribution systems

## Terms of use, privacy and license

## Dataset Version History
- V 0.1: initiate data upload

## Suggested Citation
- Please cite the following paper when you use this data hub:  
`
"PSML: A Multi-scale Time-series Dataset for Machine Learning in Decarbonized Energy Grids," Neural Information Processing System Dataset and Benchmark Track, 2021.
`\
Available at: [arXiv](https://arxiv.org/abs/XXXXXXXXXXXXXX).

- Alternative supplement dataset in Zenodo:
Available at: [Zenodo](https://zenodo.org/deposit/5130612#).

## Contact
