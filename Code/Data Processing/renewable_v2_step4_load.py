from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import requests

def load_process(file, location, year_list, save_path):
    df = pd.read_csv(file, parse_dates = True)
    df =df.loc[(df['date']>='2018-01-01') & (df['date']<='2020-12-31')]
    df = df.set_index('date')
    df_converted = pd.DataFrame(columns=['time', 'load_power'])
    hour_columns = df.columns
    for i in range(len(df.index)):
        print(df.index[i])
        for j in range(len(hour_columns)):
            date_tmp = df.index[i]
            hour_tmp = hour_columns[j]
            time_tmp = datetime.strptime(date_tmp, '%Y-%m-%d')+timedelta(hours=j)
            power_tmp = df[hour_tmp][date_tmp]
            df_converted.loc[len(df_converted.index)] = [time_tmp, power_tmp]
    power_mean = df_converted['load_power'].mean()
    
    # interpolation
    for year in year_list:
        time_esti = pd.date_range(start='1-1-'+year, end='12-31-'+year, freq='1T')
        power_esti = np.interp(time_esti, df_converted['time'], df_converted['load_power'])
        df_interp = pd.DataFrame()
        df_interp['time'] = time_esti
        df_interp['load_power'] = power_esti
        df_interp['load_power'] = df_interp['load_power']/power_mean
        df_interp.to_csv(save_path+"load_interp_"+location+'_'+year+'.csv')

if __name__=='__main__':
    # read solar generation data, rated 1 kW
    location_list = ['Houston', 'Boston', 'LosAngeles','NewYork', 'Philadelphia', 'Chicago']#['Houston', 'Boston', 'LosAngeles', 'Kansas', 'NewYork', 'Philadelphia', 'Chicago']
    state_list =['TX', 'MA', 'CA', 'NY', 'PA', 'IL']
    year_list = ['2018','2019','2020']
    load_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\load/'
    save_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\load/'
    for location in location_list:
        print(location)
        file_path = load_path+location+"_load.csv"
        load_process(file_path, location, year_list, save_path)
    



