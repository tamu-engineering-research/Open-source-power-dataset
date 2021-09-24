from cosim import cosim, simdata
import psspy as ps
#case_T = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_T\PSSE23_wind\savnw_wind_tuned.raw'
case_T = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_T\Re_wind_v3\savnw_wind_scale_down.raw'
#case_T = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_T\PSSE\IEEE_39_bus.raw'
case_D13 = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\case_D\13Bus\IEEE13Nodeckt_scaled.dss'
data_path = r'C:\Users\Dongqi Wu\OneDrive\Work\USC\data\psse_data_test.csv'

data = simdata(data_path)
env = cosim(case_T, [case_D13, case_D13], [3005, 3008], data)
print("env created")
#env.solve_ss(1)
trans,dist,info = env.dynsim(1, 800)
print(info)
