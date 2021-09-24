from cosim import cosim, simdata
import psspy as ps
import sys, os, time
import csv

#case_T = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_T\PSSE23_wind\savnw_wind_tuned.raw'
case_T = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_T\Re_wind_v3\savnw_wind_scale_down.raw'
#case_T = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_T\PSSE\IEEE_39_bus.raw'
case_D13 = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_D\13Bus\IEEE13Nodeckt_scaled.dss'
out_path = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\output_ss\pf_result_2.csv'
data_path = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\data\v3data_2.csv'

print('Reading files..')
data = simdata(data_path, version=3)
print('Start simulating!')
env = cosim(case_T, [case_D13, case_D13], [3005, 3008], data)

#end = len(env.ts)

# size of current csv
with open(out_path) as fh:
    existing_row_num = sum(1 for line in fh)
start = max(0, existing_row_num - 1)
end = start + 10000

if existing_row_num == 0:
    empty = 1
else:
    empty = 0

print('Running from row '+str(start)+' to '+str(end))
with open(out_path, 'a') as fh:
    for row in range(start, end):
        if row % 5000 == 0:
            print(str(row)+ ' lines completed')
        env.solve_ss(row)
        #time.sleep(0.2)
        # write header
        if empty:
            header = ''
            header += 'time, '
            for bus in range(env.trans.bus_num):
                header += 'Vm_'+str(env.trans.bus_ids[bus])
                header += ', '
            for bus in range(env.trans.bus_num):
                header += 'Va_'+str(env.trans.bus_ids[bus])
                header += ', '          
            for brn in range(env.trans.line_num):
                header += 'P_'+str(env.trans.line_T[brn][0])
                header += '_'+str(env.trans.line_T[brn][1])
                header += '_'+env.trans.line_ids[brn]
                header += ', '  
            for brn in range(env.trans.line_num):
                header += 'Q_'+str(env.trans.line_T[brn][0])
                header += '_'+str(env.trans.line_T[brn][1])
                header += '_'+env.trans.line_ids[brn]
                header += ', '                  
            header += '\n'
            fh.write(header)
            empty = 0
            
        # collect data
        all_vm = ps.abusreal(-1, 2, 'PU')[1][0]
        all_va = ps.abusreal(-1, 2, 'ANGLE')[1][0]
        all_p = ps.abrnreal(-1,1,3,4,1,'P')[1][0]
        all_q = ps.abrnreal(-1,1,3,4,1,'P')[1][0]

        # append
        curr_line = ''
        curr_line += env.ts[row]
        curr_line += ', '
        for bus in range(env.trans.bus_num):
            curr_line += str(all_vm[bus])
            curr_line += ', '
        for bus in range(env.trans.bus_num):
            curr_line += str(all_va[bus])
            curr_line += ', '          
        for brn in range(env.trans.line_num):
            curr_line += str(all_p[brn])
            curr_line += ', '
        for brn in range(env.trans.line_num):
            curr_line += str(all_q[brn])
            curr_line += ', '
        curr_line += '\n'
        fh.write(curr_line)
        
        env.trans.close()
        env.dist.reset()



