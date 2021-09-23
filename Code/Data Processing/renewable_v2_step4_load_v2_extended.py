from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import requests

def load_process(file, location, year_list, save_path):
    df = pd.read_csv(file, parse_dates = True,)
    flg_test = df.loc[(df['date']>='2018-01-01') & (df['date']<='2020-12-31')]
    if flg_test.empty:
        df['date_datetime'] = pd.to_datetime(df['date'])
        df['date'] = df.apply(lambda row: row['date_datetime'].strftime('%Y-%m-%d'), axis=1)
    df = df.loc[(df['date']>='2018-01-01') & (df['date']<='2020-12-31')]
    df = df.set_index('date')
    df_converted = pd.DataFrame(columns=['time', 'load_power'])
    for i in range(len(df.index)):
        print(df.index[i])
        date_tmp = df.index[i]
        hour_tmp = df['time'][i]
        time_tmp = datetime.strptime(date_tmp+" "+hour_tmp, '%Y-%m-%d %H:%M')#+timedelta(hours=j)
        power_tmp = df['load'][i]
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
        df_interp.to_csv(save_path+"load_interp_"+location+year+'.csv')

if __name__=='__main__':
    # read solar generation data, rated 1 kW
    iso_list = ['PJM',]#[ 'CAISO','NYISO','PJM','ERCOT','MISO','SPP']
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
    save_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\load_v2_extended/'
    for iso_tmp in iso_list:
        print(iso_tmp)
        iso_zone_list_tmp = iso_zone_list[iso_tmp]
        for iso_zone_tmp in iso_zone_list_tmp:
            print(iso_zone_tmp)
            file_path = load_path+iso_zone_tmp[:-1]+'.csv'
            load_process(file_path, iso_zone_tmp, year_list, save_path)
    



