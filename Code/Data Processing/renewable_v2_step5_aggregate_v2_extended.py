from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import requests


def read_data(file, location, year_list, columns, prefix):
    df_all = pd.DataFrame()
    for year in year_list:
        file_tmp = file+location+year+'.csv'
        df_tmp = pd.read_csv(file_tmp)
        df_tmp = df_tmp[columns]
        df_tmp = df_tmp.set_index('time')
        if df_all.empty:
            df_all = df_tmp.copy()
        else:
            df_all = pd.concat([df_all, df_tmp], axis=0)
    # df_all = df_all.add_prefix(prefix)
    return df_all

if __name__=='__main__':
    # read solar generation data, rated 1 kW
    iso_list = [ 'CAISO','NYISO','PJM','ERCOT','MISO','SPP']
    iso_zone_list = {
        'CAISO':['CAISO_zone_'+str(num)+"_"  for num in range(1,4+1)],
        'NYISO':['NYISO_zone_'+str(num)+"_"  for num in range(1,11+1)],
        'PJM':['PJM_zone_'+str(num)+"_"  for num in range(1,20+1)],
        'ERCOT':['ERCOT_zone_'+str(num)+"_"  for num in range(1,8+1)],
        'MISO':['MISO_zone_'+str(num)+"_"  for num in range(1,6+1)],
        'SPP':['SPP_zone_'+str(num)+"_"  for num in range(1,17+1)],
    }
    year_list = ['2018','2019','2020']
    load_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\load_v2_extended/'
    solar_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\solar_v2_extended/'
    wind_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\wind_v2_extended/'
    weather_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\weather_v2_extended/'
    save_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\psse_input_v3/'
    
    for iso_tmp in iso_list:
        print(iso_tmp)
        iso_zone_list_tmp = iso_zone_list[iso_tmp]
        for iso_zone_tmp in iso_zone_list_tmp:
            print(iso_zone_tmp)
            df_aggregate = pd.DataFrame()
            # load
            file_tmp = load_path+"load_interp_"
            prefix = ''
            df_data = read_data(file_tmp, iso_zone_tmp, year_list, ['time','load_power'], prefix)
            df_aggregate = df_data.copy()
            #wind
            file_tmp = wind_path+"wind_interp_"
            prefix = ''
            df_data = read_data(file_tmp, iso_zone_tmp, year_list, ['time','wind_power'], prefix)
            df_aggregate = pd.concat([df_aggregate, df_data],axis=1)
            #solar
            file_tmp = solar_path+"solar_interp_"
            prefix = ''
            df_data = read_data(file_tmp, iso_zone_tmp, year_list, ['time','solar_power'], prefix)
            df_aggregate = pd.concat([df_aggregate, df_data],axis=1)
            #weather
            file_tmp = weather_path+"weather_interp_"
            prefix = ''
            column_list_raw = ['time','DHI','DNI','GHI','Dew Point','Solar Zenith Angle', 'Wind Speed','Relative Humidity', 'Temperature']
            df_data = read_data(file_tmp, iso_zone_tmp, year_list, column_list_raw, prefix)
            df_aggregate = pd.concat([df_aggregate, df_data],axis=1)
            # save csv
            df_aggregate.to_csv(save_path+iso_zone_tmp+".csv")
            a=0
    



