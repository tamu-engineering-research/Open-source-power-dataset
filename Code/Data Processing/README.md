# Data Processing for Simulation
Here we describe how to process the original load consumption, weather, wind and solar time series data into the ready-for-simulation format.

## Prerequisite
- Install required packages
```angular2html
pip install -r requirements.txt
```

## Processing Details
- Before running codes, make sure the data and save paths are consistent with your downloaded data. 
- Run the codes sequentially as indicated by the name of each code file.
   - `renewable_v2_step1_weather_v2_extended.py`:
   - `renewable_v2_step2_wind_v2_extended.py`: 
   - `renewable_v2_step3_solar_v2_extended.py`: calculate the real-world zone-wide load time series data across the U.S. from 2018 to 2020 from [COVID-EMDA](https://github.com/tamu-engineering-research/COVID-EMDA.git). 
   - `renewable_v2_step4_load_v2_extended.py`: collect the real-world zone-wide load time series data across the U.S. from 2018 to 2020 from [COVID-EMDA](https://github.com/tamu-engineering-research/COVID-EMDA.git). 
   - `renewable_v2_step5_aggregate_v2_extended.py`: collect all processed files in the former steps into the ready-for-simulation format.
- The obatined results should be the same as shown in the `Minute-level Load and Renewable` folder shared in [Zenodo](https://zenodo.org/record/5130612#.YTIiZI5KiUk).

## References
1. **COVID-EMDA**:
    <em>G. Ruan, D. Wu, X. Zheng, H. Zhong, C. Kang, M. A. Dahleh, S. Sivaranjani, and L. Xie, ``A Cross-Domain Approach to Analyzing the Short-Run Impact of COVID-19 on the U.S. Electricity Sector,'' Joule, vol. 4, pp. 1-16, 2020.</em>
    https://github.com/tamu-engineering-research/COVID-EMDA.git

2. **PreREISE**:
    <em>Breakthrough Energy, PreREISE.</em>
    https://github.com/Breakthrough-Energy/PreREISE.git
    
3. **NSRDB**:
    <em>NREL, NSRDB Data Viewer.</em>
    https://maps.nrel.gov/nsrdb-viewer/?aL=mcQtmw%255Bv%255D%3Dt&bL=clight&cE=0&lR=0&mC=31.970803930433096%2C-82.705078125&zL=5
