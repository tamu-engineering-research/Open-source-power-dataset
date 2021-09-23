import numpy as np
import pandas as pd
import json
import os

##########################################################################
# ERCOT
##########################################################################
def parse_ercot_load(area, file, area_mapping=None):
    print(':: Start handling %s ...' % file)
    if 'cdr.00013101' in file:
        return _parse_ercot_load_daily_record(area, file)
    elif 'Native_Load' in file:
        return _parse_ercot_load_archive(area, file)
    print('>> WARNING: Unexpected file name!')

def _parse_ercot_load_daily_record(area, file):
    df = pd.read_csv(file, index_col='OperDay')
    df.index = pd.to_datetime(df.index)
    df['HourEnding'] = df['HourEnding'].str.replace(
        pat=r'\d\d:', repl=lambda m: np.str(int(m.group(0)[0:2]) - 1).zfill(2) + ':')  # avoid 24:00
    area_mapping = {
        'zone_1':   'COAST',
        'zone_2':   'EAST',
        'zone_3':   'FAR_WEST',
        'zone_4':   'NORTH',
        'zone_5':   'NORTH_C',
        'zone_6':   'SOUTHERN',
        'zone_7':   'SOUTH_C',
        'zone_8':   'WEST',
        }
    assert area in area_mapping.keys(), '>> WARNING: Unexpected area keyword!'
    return pd.DataFrame({
            'date': df.index.date,
            'time': df['HourEnding'],
            'load': df[area_mapping[area]].values,
        })

def _parse_ercot_load_archive(area, file):
    df = pd.read_excel(file, index_col=0)  # 'Hour Ending' or 'HourEnding'
    df.index = df.index.astype(str).str.replace(
        pat=r'\d\d:', repl=lambda m: np.str(int(m.group(0)[0:2]) - 1).zfill(2) + ':')  # avoid 24:00
    df.index = df.index.astype(str).str.replace('DST', '')
    df.index = pd.to_datetime(df.index)
    area_mapping = {
        'zone_1':   'COAST',
        'zone_2':   'EAST',
        'zone_3':   'FWEST',
        'zone_4':   'NORTH',
        'zone_5':   'NCENT',
        'zone_6':   'SOUTH',
        'zone_7':   'SCENT',
        'zone_8':   'WEST',
        }
    return pd.DataFrame({
        'date': df.index.date,
        'time': df.index.strftime('%H:%M'),
        'load': df[area_mapping[area]].values,
    })

##########################################################################
# PJM
##########################################################################
def parse_pjm_load(area, file):
    print(':: Start handling %s ...' % file)
    df = pd.read_csv(file, index_col='datetime_beginning_ept')
    df.index = pd.to_datetime(df.index)
    area_mapping = {
        'zone_1': 'AE', 
        'zone_2': 'AEP',  
        'zone_3': 'AP',
        'zone_4': 'ATSI',
        'zone_5': 'BC', 
        'zone_6': 'CE',  
        'zone_7': 'DAY',
        'zone_8': 'DEOK',
        'zone_9': 'DOM', 
        'zone_10': 'DPL',  
        'zone_11': 'DUQ',
        'zone_12': 'EKPC',
        'zone_13': 'JC', 
        'zone_14': 'ME',  
        'zone_15': 'PE',
        'zone_16': 'PEP',
        'zone_17': 'PL', 
        'zone_18': 'PN',  
        'zone_19': 'PS',
        'zone_20': 'RECO',
    }
    assert area in area_mapping.keys(), '>> WARNING: Unexpected area keyword!'
    dfsel = df[df['zone'] == area_mapping[area]].loc[:, 'mw']
    df_zone = df[df['zone'] == area_mapping[area]].groupby(by='datetime_beginning_ept').sum()['mw']
    return pd.DataFrame({
        'date': df_zone.index.date,
        'time': df_zone.index.strftime('%H:%M'),
        'load': df_zone.values,
    })

##########################################################################
# SPP
##########################################################################
def parse_spp_load(area, file):
    print(':: Start handling %s ...' % file)
    df = pd.read_csv(file, index_col='MarketHour')
    df.index = pd.to_datetime(df.index)
    df.index = df.index - pd.Timedelta(hours=6) #removed tz_localize('GMT') after index
    area_mapping = {
        'zone_1': ' CSWS', 
        'zone_2': ' EDE',  
        'zone_3': ' GRDA',
        'zone_4': ' INDN',
        'zone_5': ' KACY', 
        'zone_6': ' KCPL',  
        'zone_7': ' LES',
        'zone_8': ' MPS',
        'zone_9': ' NPPD', 
        'zone_10': ' OKGE',  
        'zone_11': ' OPPD',
        'zone_12': ' SECI',
        'zone_13': ' SPRM', 
        'zone_14': ' SPS',  
        'zone_15': ' WAUE',
        'zone_16': ' WFEC',
        'zone_17': ' WR', 
    }
    assert area in area_mapping.keys(), '>> WARNING: Unexpected area keyword!'
    dfsel = df[area_mapping[area]]
    if len(dfsel.shape) > 1:
        dfsel = dfsel.sum(axis=1)
    return pd.DataFrame({
        'date': dfsel.index.date,
        'time': dfsel.index.strftime('%H:%M'),
        'load': dfsel.values,
    })
##########################################################################
# NYISO
##########################################################################
def parse_nyiso_load(area, file):
    print(':: Start handling %s ...' % file)
    df = pd.read_csv(file, index_col='Time Stamp')
    df.index = pd.to_datetime(df.index)
    # assert area in ['rto', 'nyc'], '>> WARNING: Unexpected area keyword!'
    area_mapping = {
        'zone_1': 'CAPITL', 
        'zone_2': 'CENTRL',  
        'zone_3': 'DUNWOD',
        'zone_4': 'GENESE',
        'zone_5': 'HUD VL', 
        'zone_6': 'LONGIL',  
        'zone_7': 'MHK VL',
        'zone_8': 'MILLWD',
        'zone_9': 'N.Y.C.', 
        'zone_10': 'NORTH',  
        'zone_11': 'WEST',
    }
    dfsel = df[df['Name'] == area_mapping[area]].loc[:, 'Integrated Load']
    return pd.DataFrame({
        'date': dfsel.index.date,
        'time': dfsel.index.strftime('%H:%M'),
        'load': dfsel.values,
    })
##########################################################################
# MISO
##########################################################################
def parse_miso_load(area, file):
    print(':: Start handling %s ...' % file)
    df = pd.read_excel(file, skiprows=[0, 1, 2, 3, 5], skipfooter=27)
    area_mapping = {
        'zone_1':  'LRZ1 ActualLoad (MWh)',
        'zone_2':  'LRZ2_7 ActualLoad (MWh)',
        'zone_3':  'LRZ3_5 ActualLoad (MWh)',
        'zone_4':  'LRZ4 ActualLoad (MWh)',
        'zone_5':  'LRZ6 ActualLoad (MWh)',
        'zone_6':  'LRZ8_9_10 ActualLoad (MWh)',
    }
    assert area in area_mapping.keys(), '>> WARNING: Unexpected area keyword!'
    dfsel = df[area_mapping[area]]
    if len(dfsel.shape) > 1:
        dfsel = dfsel.sum(axis=1)
    return pd.DataFrame({
        'date': pd.to_datetime(df['Market Day']).dt.date,
        'time': (df['HourEnding'] - 1).astype(str).str.zfill(2) + ':00',
        'load': dfsel.values,
    })
##########################################################################
# CAISO
##########################################################################
def parse_caiso_load(area, file, after_date):
    print(':: Start handling %s ...' % file)
    # assert area in ['rto', 'la'], '>> WARNING: Unexpected area keyword!'
    area_mapping = {
        'zone_1':  'TAC_ECNTR',
        'zone_2':  'TAC_NCNTR',
        'zone_3':  'TAC_NORTH',
        'zone_4':  'TAC_SOUTH',
        'zone_5':  'Los_Angeles',
    }
    if area_mapping[area]=='Los_Angeles' and 'Demand_for_Los_Angeles' in file:
        return _parse_ca_load_la(file)
    if (not area_mapping[area]=='Los_Angeles') and 'ENE_SLRS_DAM' in file:
        return _parse_caiso_load_rto(file, area_mapping[area])
    print('>> WARNING: Dismatch area keyword & file name! Please double check!')

def _parse_caiso_load_rto(file, area):
    df = pd.read_csv(file, index_col='INTERVALSTARTTIME_GMT')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.index = df.index.tz_convert('America/Los_Angeles') #removed .tz_localize('GMT') after index
    dfsel = df[df['TAC_ZONE_NAME'] == area]
    return pd.DataFrame({
        'date': dfsel.index.date,
        'time': dfsel.index.strftime('%H:%M'),
        'load': dfsel['MW'],
    })

# def _parse_ca_load_la(file, after_date=None):
#     with open(file) as fn:
#         dict = json.load(fn)
#     df = pd.DataFrame(dict['series'][0]['data'])
#     df.columns = ['date_time', 'load']
#     df.index = pd.to_datetime(df['date_time'], utc=True)
#     df.index = df.index.tz_convert('America/Los_Angeles') #removed tz_localize('GMT') after index
#     if after_date is not None:
#         df = df.loc[df.index.date >= after_date.date()]
#     return pd.DataFrame({
#         'date': df.index.date,
#         'time': df.index.strftime('%H:%M'),
#         'load': df['load'],
#     })

##########################################################################
# ISONE
##########################################################################
def parse_isone_load(area, file):
    print(':: Start handling %s ...' % file)
    assert area in ['rto', 'boston'], '>> WARNING: Unexpected area keyword!'
    if area == 'rto' and 'rt_hourlysysload' in file:
        return _parse_isone_load_rto(file)
    if area == 'boston' and 'OI_darthrmwh_iso' in file:
        return _parse_isone_load_boston(file)
    print('>> WARNING: Dismatch area keyword & file name! Please double check!')

def _parse_isone_load_rto(file):
    df = pd.read_csv(file, skiprows=[0, 1, 2, 3, 5], skipfooter=1, engine='python').drop(columns='H')
    if df['Hour Ending'].dtype.kind not in 'iuf':  # not number
        print(':::: Find non-numeric hour records in %s' % file)
        df = df[df['Hour Ending'].astype(str).apply(lambda x: x.replace('.', '').isnumeric())]  # remove records 02X
        df['Hour Ending'] = df['Hour Ending'].astype(int)
    return pd.DataFrame({
        'date': pd.to_datetime(df['Date']).dt.date,
        'time': (df['Hour Ending'] - 1).astype(str).str.zfill(2) + ':00',
        'load': df['Total Load'],
    })

def _parse_isone_load_boston(file):
    df = pd.read_csv(file, skiprows=[0, 1, 2, 3, 4, 6], skipfooter=1, engine='python')
    df.columns = ['H', 'Date', 'Hour Ending', 'Day Ahead', 'Real Time']
    df = df.drop(columns=['H', 'Day Ahead'])
    if df['Hour Ending'].dtype.kind not in 'iuf':  # not number
        print(':::: Find non-numeric hour records in %s' % file)
        df = df[df['Hour Ending'].astype(str).apply(lambda x: x.replace('.', '').isnumeric())]  # remove records 02X
        df['Hour Ending'] = df['Hour Ending'].astype(int)
    return pd.DataFrame({
        'date': pd.to_datetime(df['Date']).dt.date,
        'time': (df['Hour Ending'] - 1).astype(str).str.zfill(2) + ':00',
        'load': df['Real Time'],
    }).dropna()


if __name__=="__main__":
    read_root_path = r'C:\Users\zheng\OneDrive\Documents\GitHub\COVID-EMDA\data_source/'
    save_path = r'C:\Users\zheng\Google Drive\Colab Notebooks\94 PowerDataSet\Dataset\Renewable\load_v2_extended/'
    RTO_name = 'PJM'#'MISO'#'NYISO'#'SPP'#'PJM'#'ERCOT'
    if RTO_name=="ERCOT":
        read_path = read_root_path+"ercot/load/"
        area_list = ['zone_'+str(num) for num in range(1,9,1)]
        for area_tmp in area_list:
            print(area_tmp)
            data_area = pd.DataFrame()
            for path, subdirs, files in os.walk(read_path):
                for name in files:
                    file_tmp = os.path.join(path, name)
                    print(file_tmp)
                    data_area_tmp = parse_ercot_load(area=area_tmp, file=file_tmp,)
                    if data_area.empty:
                        data_area = data_area_tmp.copy()
                    else:
                        data_area = pd.concat([data_area, data_area_tmp.copy()], axis=0, ignore_index=True)
            data_area = data_area.sort_values(['date','time'], ascending=(True, True))
            data_area = data_area.reset_index(drop=True)
            data_area.to_csv(save_path+RTO_name+"_"+area_tmp+'.csv')
            a=0
    elif RTO_name=="PJM":
        read_path = read_root_path+"pjm/load/"
        area_list = ['zone_'+str(num) for num in range(1,21,1)]
        for area_tmp in area_list:
            print(area_tmp)
            data_area = pd.DataFrame()
            for path, subdirs, files in os.walk(read_path):
                for name in files:
                    file_tmp = os.path.join(path, name)
                    print(file_tmp)
                    data_area_tmp = parse_pjm_load(area=area_tmp, file=file_tmp,)
                    if data_area.empty:
                        data_area = data_area_tmp.copy()
                    else:
                        data_area = pd.concat([data_area, data_area_tmp.copy()], axis=0, ignore_index=True)
            data_area = data_area.sort_values(['date','time'], ascending=(True, True))
            data_area = data_area.reset_index(drop=True)
            data_area.to_csv(save_path+RTO_name+"_"+area_tmp+'.csv')
    elif RTO_name=="SPP":
        read_path = read_root_path+"spp/load/"
        area_list = ['zone_'+str(num) for num in range(1,18,1)]
        for area_tmp in area_list:
            print(area_tmp)
            data_area = pd.DataFrame()
            for path, subdirs, files in os.walk(read_path):
                for name in files:
                    file_tmp = os.path.join(path, name)
                    print(file_tmp)
                    data_area_tmp = parse_spp_load(area=area_tmp, file=file_tmp,)
                    if data_area.empty:
                        data_area = data_area_tmp.copy()
                    else:
                        data_area = pd.concat([data_area, data_area_tmp.copy()], axis=0, ignore_index=True)
            data_area = data_area.sort_values(['date','time'], ascending=(True, True))
            data_area = data_area.reset_index(drop=True)
            data_area.to_csv(save_path+RTO_name+"_"+area_tmp+'.csv')
    elif RTO_name=="NYISO":
        read_path = read_root_path+"nyiso/load/"
        area_list = ['zone_'+str(num) for num in range(1,12,1)]
        for area_tmp in area_list:
            print(area_tmp)
            data_area = pd.DataFrame()
            for path, subdirs, files in os.walk(read_path):
                for name in files:
                    file_tmp = os.path.join(path, name)
                    print(file_tmp)
                    data_area_tmp = parse_nyiso_load(area=area_tmp, file=file_tmp,)
                    if data_area.empty:
                        data_area = data_area_tmp.copy()
                    else:
                        data_area = pd.concat([data_area, data_area_tmp.copy()], axis=0, ignore_index=True)
            data_area = data_area.sort_values(['date','time'], ascending=(True, True))
            data_area = data_area.reset_index(drop=True)
            data_area.to_csv(save_path+RTO_name+"_"+area_tmp+'.csv')
    elif RTO_name=="MISO":
        read_path = read_root_path+"miso/load/"
        area_list = ['zone_'+str(num) for num in range(1,7,1)]
        for area_tmp in area_list:
            print(area_tmp)
            data_area = pd.DataFrame()
            for path, subdirs, files in os.walk(read_path):
                for name in files:
                    file_tmp = os.path.join(path, name)
                    print(file_tmp)
                    data_area_tmp = parse_miso_load(area=area_tmp, file=file_tmp,)
                    if data_area.empty:
                        data_area = data_area_tmp.copy()
                    else:
                        data_area = pd.concat([data_area, data_area_tmp.copy()], axis=0, ignore_index=True)
            data_area = data_area.sort_values(['date','time'], ascending=(True, True))
            data_area = data_area.reset_index(drop=True)
            data_area.to_csv(save_path+RTO_name+"_"+area_tmp+'.csv')
    elif RTO_name=="CAISO":
        read_path = read_root_path+"caiso/load/"
        area_list = ['zone_'+str(num) for num in range(5,6,1)]
        for area_tmp in area_list:
            print(area_tmp)
            data_area = pd.DataFrame()
            for path, subdirs, files in os.walk(read_path):
                for name in files:
                    file_tmp = os.path.join(path, name)
                    print(file_tmp)
                    data_area_tmp = parse_caiso_load(area=area_tmp, file=file_tmp, after_date=None)
                    if not (data_area_tmp is None):
                        if data_area.empty:
                            data_area = data_area_tmp.copy()
                        else:
                            data_area = pd.concat([data_area, data_area_tmp.copy()], axis=0, ignore_index=True)
            data_area = data_area.sort_values(['date','time'], ascending=(True, True))
            data_area = data_area.reset_index(drop=True)
            data_area.to_csv(save_path+RTO_name+"_"+area_tmp+'.csv')
    elif RTO_name=="ISONE":
        read_path = read_root_path+"isone/load/"
        area_list = ['zone_'+str(num) for num in range(5,6,1)]
        for area_tmp in area_list:
            print(area_tmp)
            data_area = pd.DataFrame()
            for path, subdirs, files in os.walk(read_path):
                for name in files:
                    file_tmp = os.path.join(path, name)
                    print(file_tmp)
                    data_area_tmp = parse_isone_load(area=area_tmp, file=file_tmp, after_date=None)
                    if not (data_area_tmp is None):
                        if data_area.empty:
                            data_area = data_area_tmp.copy()
                        else:
                            data_area = pd.concat([data_area, data_area_tmp.copy()], axis=0, ignore_index=True)
            data_area = data_area.sort_values(['date','time'], ascending=(True, True))
            data_area = data_area.reset_index(drop=True)
            data_area.to_csv(save_path+RTO_name+"_"+area_tmp+'.csv')
