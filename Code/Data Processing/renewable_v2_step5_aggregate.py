from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import requests


def read_data(file, location, year_list, columns, prefix):
    df_all = pd.DataFrame()
    for year in year_list:
        file_tmp = file+location+"_"+year+'.csv'
        df_tmp = pd.read_csv(file_tmp)
        df_tmp = df_tmp[columns]
        df_tmp = df_tmp.set_index('time')
        if df_all.empty:
            df_all = df_tmp.copy()
        else:
            df_all = pd.concat([df_all, df_tmp], axis=0)
    df_all = df_all.add_prefix(prefix)
    return df_all

if __name__=='__main__':
    # read solar generation data, rated 1 kW
    location_list = ['Houston', 'Boston', 'LosAngeles', 'NewYork', 'Philadelphia', 'Chicago']#['Houston', 'Boston', 'LosAngeles', 'Kansas', 'NewYork', 'Philadelphia', 'Chicago']
    state_list =['TX', 'MA', 'CA', 'NY', 'PA', 'IL']
    year_list = ['2018','2019','2020']
    load_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\load/'
    solar_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\solar/'
    wind_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\wind/'
    weather_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\weather/'
    save_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable/'
    
    df_aggregate = pd.DataFrame()
    #load
    for location_id in range(len(location_list)):
        print(location_id)
        file_tmp = load_path+'load_interp_'
        location = location_list[location_id]
        prefix = 'P'+str(location_id)+"_"
        df_data = read_data(file_tmp, location, year_list, ['time','load_power'], prefix)
        if df_aggregate.empty:
            df_aggregate = df_data.copy()
        else:
            df_aggregate = pd.concat([df_aggregate, df_data],axis=1)
    #wind
    for location_id in range(2):
        print(location_id)
        file_tmp = wind_path+'wind_interp_'
        location = location_list[location_id]
        prefix = 'P'+str(location_id)+"_"
        df_data = read_data(file_tmp, location, year_list, ['time','wind_power'], prefix)
        df_aggregate = pd.concat([df_aggregate, df_data],axis=1)
    #solar
    for location_id in range(2):
        print(location_id)
        file_tmp = solar_path+'solar_interp_'
        location = location_list[location_id]
        prefix = 'P'+str(location_id)+"_"
        df_data = read_data(file_tmp, location, year_list, ['time','solar_power'], prefix)
        df_aggregate = pd.concat([df_aggregate, df_data],axis=1)
    #weather
    for location_id in range(len(location_list)):
        print(location_id)
        file_tmp = weather_path+'weather_interp_'
        location = location_list[location_id]
        prefix = 'P'+str(location_id)+"_"
        column_list_raw = ['time','DHI','DNI','GHI','Dew Point','Solar Zenith Angle', 'Wind Speed','Relative Humidity', 'Temperature']
        df_data = read_data(file_tmp, location, year_list, column_list_raw, prefix)
        df_aggregate = pd.concat([df_aggregate, df_data],axis=1)

    df_aggregate.to_csv(save_path+'psse_input_v2.csv')
    



