from powersimdata.input.grid import Grid
# from PreREISE import prereise as prr#rap, impute, helpers
from prereise.gather.winddata.rap import rap, impute, helpers
from prereise.gather.winddata.rap import power_curves
from datetime import datetime

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
    weather_raw.to_csv(save_path+'weather_'+location+"_"+year+'.csv')
    weather_interp.to_csv(save_path+'weather_interp_'+location+"_"+year+'.csv')

if __name__=='__main__':

    # read solar generation data, rated 1 kW
    location_list = [ 'NewYork', 'Philadelphia', 'Chicago']#['Houston', 'Boston', 'LosAngeles', 'Kansas', 'NewYork', 'Philadelphia', 'Chicago']
    year_list = ['2018','2019','2020']
    weahter_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\solar\NREL_PSM_v3_5min_'
    save_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\weather/'
    for location in location_list:
        for year in year_list:
            print(location+'_'+year)
            solar_data = read_solar_weather_csv(
                location=location, year=year, 
                file_name = weahter_path+location+'/'+location+"_"+year+".csv",
                save_path=save_path)
            



