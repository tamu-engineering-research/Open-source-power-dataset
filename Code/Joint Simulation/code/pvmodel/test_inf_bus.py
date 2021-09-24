import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from solar_inverter import solar_inverter

def inf_bus_simu(V_inf_D, V_inf_Q, V_mag, V_ang):
    # V_inf_D = 379
    # V_inf_Q = -20
    w_n = 50*2*np.pi
    r = 0.35+0.03
    L = 0.58/w_n+0.35e-3
    Z = np.array([[r, -w_n*L],[w_n*L, r]])
    Z_inv = inv(Z)
    V_D = V_mag*np.cos(V_ang)
    V_Q = V_mag*np.sin(V_ang)
    I_D = Z_inv[0,0]*(V_D - V_inf_D) + Z_inv[0,1]*(V_Q - V_inf_Q)
    I_Q = Z_inv[1,0]*(V_D - V_inf_D) + Z_inv[1,1]*(V_Q - V_inf_Q)
    I_mag = np.sqrt(I_D**2+I_Q**2)
    I_ang = np.angle(I_D + 1j *I_Q)
    return I_mag, I_ang

# parameter setting of solar inverter
para = {}
para['dt'] = 0.0001 #### simulation step size
para['Sb'] = 10e3 #### nominal power
para['Vb'] = 220*np.sqrt(3)#### nominal voltage
para['f'] = 50
para['w_n'] = para['f']*2*np.pi
para['L_f'] =1.35e-3
para['C_f'] =50e-6
para['r_f'] =0.1
para['w_c'] = 31.41
para['K_pv'] = 0.05
para['K_iv'] = 390
para['K_pc'] = 10.5
para['K_ic'] = 16e3
para['F'] = 0.75
para['Tf'] = 1.5*para['Sb']
para['Ta'] = 1
para['Tv'] = 10
para['Df'] = 1.5*para['Sb']
para['Da'] = 0.2/para['Sb']
para['Dv'] = 0.2/7*para['Vb']/para['Sb']

# simulation first to get initial V and I
# P = XXX #################
# Q = XXX #################
V_mag = 381.8
V_ang = 0.1
V_inf_D = 379
V_inf_Q = -20
I_mag, I_ang = inf_bus_simu(V_inf_D, V_inf_Q,  V_mag, V_ang) ################# equivalent PQ bus

# initialize class solar inverter
system_13bus_solar = solar_inverter( I_mag, I_ang, V_mag, V_ang, para=para)

# start dynamical simulation
for i in range(2000):
    print(i)
    # update internal state
    system_13bus_solar.cal_next_step()
    # update new terminal voltage
    V_mag, V_ang = system_13bus_solar.get_latest_terminal_voltage()
    # create fault
    if (i>=20) & (i<=int(0.01/para['dt'])):
        V_inf_D = 379*0.8
        V_inf_Q = -20*1.2
    else:
        V_inf_D = 379
        V_inf_Q = -20
    # get corresponding new terminal current
    I_mag, I_ang = inf_bus_simu(V_inf_D, V_inf_Q, V_mag, V_ang)
    # update some setting variables
    system_13bus_solar.cal_next_step_algebraic(I_mag, I_ang)

delta = system_13bus_solar.time_varying_state['delta']
v_od = system_13bus_solar.time_varying_state['v_od']
v_oq = system_13bus_solar.time_varying_state['v_oq']
w = system_13bus_solar.time_varying_state['w']
plt.figure()
plt.plot(delta)
plt.show()

a=0
