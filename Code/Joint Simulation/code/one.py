from cosim import cosim, simdata
import psspy as ps
import sys, os
import gc
import csv

ep = int(sys.argv[1])


#case_T = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_T\PSSE23_wind\savnw_wind_tuned.raw'
case_T = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_T\Re_wind_v3\savnw_wind_scale_down.raw'
#case_T = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_T\PSSE\IEEE_39_bus.raw'
case_D13 = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_D\13Bus\IEEE13Nodeckt_scaled.dss'
data_path = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\data\psse_data_out.csv'
out_path = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\output_fig'

#env.solve_ss(1)
sec_num = 4
data = simdata(data_path)

env = cosim(case_T, [case_D13, case_D13], [3005, 3008], data)
step_num = int(sec_num / env.dyn_ts)
# find a row in the data file
row_idx = ep
# simulate
trans_ts, dist_ts, info = env.dynsim(row_idx, step_num)

# create path
epi_dir = out_path + '\\row_' + str(row_idx)
if not os.path.exists(epi_dir):
    os.mkdir(epi_dir)
trans_out_path = epi_dir + '\\trans.csv'
dist_out_path = epi_dir + '\\dist.csv'
info_out_path = epi_dir + '\\info.csv'


## trans
keys, vals = [], []
for key, val in trans_ts.items():
    keys.append(key)
    vals.append(val)
keys_num = len(keys)
# process header and value array
with open(trans_out_path, 'w') as fh:
    # header
    line = ''
    line += keys[-1]
    for k in range(keys_num-1):
        line += ', '
        line += keys[k]
    line += '\n'
    fh.write(line)
    # data
    for t in range(step_num):
        line = ''
        line += str(vals[-1][t])
        for k in range(keys_num-1):
            line += ', '
            line += str(vals[k][t])
        line += '\n'
        fh.write(line)
del keys
del val
gc.collect()

## dist
keys, vals = [], []
for key, val in dist_ts.items():
    keys.append(key)
    vals.append(val)
keys_num = len(keys)
# process header and value array
with open(dist_out_path, 'w') as fh:
    # header
    line = ''
    for k in range(keys_num-1):
        line += keys[k]
        line += ', '
    line += keys[keys_num-1]
    line += '\n'
    fh.write(line)
    # data
    for t in range(step_num):
        line = ''
        for k in range(keys_num-1):
            line += str(vals[k][t])
            line += ', '
        line += str(vals[-1][t])
        line += '\n'
        fh.write(line)
del keys
del val
del env
gc.collect()

# info
with open(info_out_path, 'w') as fh:
    for key, val in info.items():
        fh.write(key +', '+ str(val) + '\n')

