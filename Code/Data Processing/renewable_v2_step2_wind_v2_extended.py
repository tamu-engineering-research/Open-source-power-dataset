from powersimdata.input.grid import Grid
# from PreREISE import prereise as prr#rap, impute, helpers
from prereise.gather.winddata.rap import rap, impute, helpers
from prereise.gather.winddata.rap import power_curves
from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import requests

def read_solar_weather_csv(
    location, year, power_curve,
    file_name, save_path,
    ASOS_height = 10,
    turbine_height = 80,
    alpha=0.15):
    data = pd.read_csv(file_name, delimiter = ',')
    data['time'] = data.apply(lambda row: datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S'), axis=1)
    column_list = ['time', 'Wind Speed']
    weather = data[['time', 'Wind Speed']].copy()
    weather['Wind Speed_noise'] = weather.apply(lambda row: np.maximum(0, row['Wind Speed']+np.random.normal(0, 0.05, 1)[0]), axis=1)
    weather['Wind Speed_turbine'] = weather['Wind Speed_noise']*(turbine_height/ASOS_height)**0.15
    # estimate wind energy based on the speed
    wind_speed_base = power_curve.index.to_numpy()
    power_base  = power_curve.to_numpy()
    wind_speed_esti = weather['Wind Speed_turbine'].to_numpy()
    estimated_power = np.interp(wind_speed_esti, wind_speed_base, power_base)
    weather['wind_power'] = estimated_power
    # interpolation
    wind_raw = weather[['time','wind_power']].copy()
    wind_interp = pd.DataFrame()
    time_interp = pd.date_range(start='1-1-'+year, end='12-31-'+year, freq='1T')
    wind_interp['time'] = time_interp
    wind_interp['wind_power'] = np.interp(time_interp, wind_raw['time'].copy(), wind_raw['wind_power'].copy())
    # save csv
    wind_raw.to_csv(save_path+'wind_'+location+year+'.csv')
    wind_interp.to_csv(save_path+'wind_interp_'+location+year+'.csv')


if __name__=='__main__':
    # read power curves utilizing the functions in PreREISE
    state_power_curves = power_curves.get_state_power_curves()
    turbine_power_curves = power_curves.get_turbine_power_curves()

    # read solar generation data, rated 1 kW
    iso_list = [ 'CAISO','NYISO','PJM','ERCOT','MISO','SPP']
    state_list = ['CA','NY','IL','TX','OK','MI']
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
    save_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\wind_v2_extended/'
    i=0
    for iso_tmp in iso_list:
        print(iso_tmp)
        state = state_list[i]
        i+=1
        iso_zone_list_tmp = iso_zone_list[iso_tmp]
        for iso_zone_tmp in iso_zone_list_tmp:
            for year in year_list:
                file_name = 'weather_'+iso_zone_tmp+year+'.csv'
                print(iso_zone_tmp+'_'+year)
                solar_data = read_solar_weather_csv(
                    location=iso_zone_tmp, year=year, 
                    power_curve=state_power_curves[state],
                    file_name = weather_path+'weather_'+iso_zone_tmp+year+".csv",
                    save_path=save_path)
            



