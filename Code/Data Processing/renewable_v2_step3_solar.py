from powersimdata.input.grid import Grid
# from PreREISE import prereise as prr#rap, impute, helpers
from prereise.gather.winddata.rap import rap, impute, helpers
from prereise.gather.winddata.rap import power_curves
from datetime import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import requests

def read_solar_csv(
    location, year,
    file_name, save_path,):
    # read data
    data = pd.read_csv(file_name, delimiter = ',')
    data['time'] = data.apply(lambda row: datetime.strptime(row['Time stamp'], '%b %d, %I:%M %p'), axis=1)
    data['time'] = data.apply(lambda row: row['time'].replace(year=int(year)), axis=1)
    data['solar_power'] = data['System power generated | (kW)'].copy()/4 # 4 is maximum DC power
    # interpolation
    solar_raw = data.copy()
    time_esti = pd.date_range(start='1-1-'+year, end='12-31-'+year, freq='1T')
    power_esti = np.interp(time_esti, solar_raw['time'], solar_raw['solar_power'])
    solar_interp = pd.DataFrame()
    solar_interp['time'] = time_esti
    solar_interp['solar_power'] = power_esti
    # save csv
    solar_raw.to_csv(save_path+'solar_'+location+"_"+year+'.csv')
    solar_interp.to_csv(save_path+'solar_interp_'+location+"_"+year+'.csv')


if __name__=='__main__':
    # read power curves utilizing the functions in PreREISE
    state_power_curves = power_curves.get_state_power_curves()
    turbine_power_curves = power_curves.get_turbine_power_curves()

    # read solar generation data, rated 1 kW
    location_list = ['Houston', 'Boston', 'LosAngeles','NewYork', 'Philadelphia', 'Chicago']#['Houston', 'Boston', 'LosAngeles', 'Kansas', 'NewYork', 'Philadelphia', 'Chicago']
    state_list =['TX', 'MA', 'CA', 'NY', 'PA', 'IL']
    year_list = ['2018','2019','2020']
    weather_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\solar/NREL_PSM_v3_5min_'
    save_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\solar/'
    for location_id in range(len(location_list)):
        location = location_list[location_id]
        state = state_list[location_id]
        for year in year_list:
            print(location+'_'+year)
            solar_data = read_solar_csv(
                location=location, year=year, 
                file_name = weather_path+location+'/solar_'+location+"_"+year+".csv",
                save_path=save_path)
            



