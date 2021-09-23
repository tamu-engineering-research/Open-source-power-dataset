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
    solar_raw.to_csv(save_path+'solar_'+location+year+'.csv')
    solar_interp.to_csv(save_path+'solar_interp_'+location+year+'.csv')


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
    solar_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\solar_v2_extended/solar_SAM_output/'
    save_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\solar_v2_extended/'
    for iso_tmp in iso_list:
        print(iso_tmp)
        iso_zone_list_tmp = iso_zone_list[iso_tmp]
        for iso_zone_tmp in iso_zone_list_tmp:
            for year in year_list:
                file_name = iso_zone_tmp+year+'.csv'
                print(iso_zone_tmp+year)
                solar_data = read_solar_csv(
                    location=iso_zone_tmp, year=year, 
                    file_name = solar_path+iso_zone_tmp+year+".csv",
                    save_path=save_path)
            



