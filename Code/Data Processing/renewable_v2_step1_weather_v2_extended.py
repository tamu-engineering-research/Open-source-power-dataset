from powersimdata.input.grid import Grid
# from PreREISE import prereise as prr#rap, impute, helpers
from prereise.gather.winddata.rap import rap, impute, helpers
from prereise.gather.winddata.rap import power_curves
from datetime import datetime
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import requests

def read_solar_weather_csv(location, year, file_name, save_path):
    data = pd.read_csv(file_name, delimiter = ',', skiprows=2)
    data['time_str'] = data.apply(lambda row: str(int(row['Year']))+"-"+str(int(row['Month']))+"-"+str(int(row['Day']))+' '+str(int(row['Hour']))+':'+str(int(row['Minute'])), axis=1)
    data['time'] = data.apply(lambda row: datetime.strptime(row['time_str'], '%Y-%m-%d %H:%M'), axis=1)
    column_list_raw = ['time','DHI','DNI','GHI','Dew Point','Solar Zenith Angle', 'Wind Speed','Relative Humidity', 'Temperature']
    weather_raw = data[column_list_raw].copy()
    weather_interp = pd.DataFrame()
    time_interp = pd.date_range(start='1-1-'+year, end='12-31-'+year, freq='1T')
    weather_interp['time'] = time_interp
    for i in range(1, len(column_list_raw)):
        column_tmp = column_list_raw[i]
        column_interp = np.interp(time_interp, data['time'].copy(), data[column_tmp].copy())
        weather_interp[column_tmp] = column_interp
    weather_raw.to_csv(save_path+'weather_'+location+year+'.csv')
    weather_interp.to_csv(save_path+'weather_interp_'+location+year+'.csv')

if __name__=='__main__':
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
    weather_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\weather_v2_extended/'
    save_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\weather_v2_extended/'
    # folder_list = [x[0] for x in os.walk(weather_path)]
    # folder_list = folder_list[1:]
    folder_list = next(os.walk(weather_path))[1]
    for iso_tmp in iso_list:
        print(iso_tmp)
        iso_zone_list_tmp = iso_zone_list[iso_tmp]
        for iso_zone_tmp in iso_zone_list_tmp:
            for folder in folder_list:
                if folder.startswith(iso_zone_tmp):
                    subfolder  = weather_path+folder
                    for year in year_list:
                        print(iso_zone_tmp+year)
                        for root, direct, files in os.walk(subfolder+'/'):
                            for file in files:
                                if file.endswith(year+'.csv'):
                                    solar_data = read_solar_weather_csv(
                                        location=iso_zone_tmp, year=year, 
                                        file_name = subfolder+'/'+file,
                                        save_path=save_path)
            



