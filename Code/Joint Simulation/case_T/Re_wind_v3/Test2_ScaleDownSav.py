'''
Please build with py27: Tools --> Build System --> py27
'''
# Incrementally decrease the load
import pssexplore34
import psspy
from psspy import _i, _f
import redirect
psspy.psseinit()
import dyntools
import numpy as np
import pssplot
psspy.psseinit()

psspy.case(r"""savnw_wind.sav""")


gen_P = np.array([750.0, 750.0, 800.0, 600.0, 258.656, 100.0])
gen_Q = np.array([91.5, 91.5,593.2, 70.7, 67.0, 0.0])





load_P = np.array([200.0000, 600.0000, 400.0000, 300.0000, 1200.0000, 100.0000, 200.0000, 200.0000])
load_Q = np.array([100.0000, 450.0000, 350.0000, 150.0000, 700.0000, 50.0000, 75.0000, 75.0000])

perc = 0.0005 # change P and Q "perc" every time

for k in np.arange(0.75, 0.74, -perc):
	gen_P_each = gen_P*k
	gen_Q_each = gen_Q*k
	load_P_each = load_P*k
	load_Q_each = load_Q*k
	# modify P and Q
	# change Pgen
	psspy.machine_chng_2(101,r"""1""",[_i,_i,_i,_i,_i,_i],[ gen_P_each[0],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.machine_chng_2(102,r"""1""",[_i,_i,_i,_i,_i,_i],[ gen_P_each[1],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.machine_chng_2(206,r"""1""",[_i,_i,_i,_i,_i,_i],[ gen_P_each[2],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.machine_chng_2(211,r"""1""",[_i,_i,_i,_i,_i,_i],[ gen_P_each[3],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.machine_chng_2(3011,r"""1""",[_i,_i,_i,_i,_i,_i],[ gen_P_each[4],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.machine_chng_2(3018,r"""1""",[_i,_i,_i,_i,_i,_i],[ gen_P_each[5],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	# change Qgen
	psspy.machine_chng_2(101,r"""1""",[_i,_i,_i,_i,_i,_i],[_f, gen_Q_each[0],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.machine_chng_2(102,r"""1""",[_i,_i,_i,_i,_i,_i],[_f, gen_Q_each[1],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.machine_chng_2(206,r"""1""",[_i,_i,_i,_i,_i,_i],[_f, gen_Q_each[2],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.machine_chng_2(211,r"""1""",[_i,_i,_i,_i,_i,_i],[_f, gen_Q_each[3],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	psspy.machine_chng_2(3011,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,gen_Q_each[4],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	# Q at 3018 is zero 
	#psspy.machine_chng_2(3018,r"""1""",[_i,_i,_i,_i,_i,_i],[_f,gen_Q_each[5],_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
	# change P load
	psspy.load_chng_5(153,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ load_P_each[0],_f,_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(154,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ load_P_each[1],_f,_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(154,r"""2""",[_i,_i,_i,_i,_i,_i,_i],[ load_P_each[2],_f,_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(203,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ load_P_each[3],_f,_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(205,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ load_P_each[4],_f,_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(3005,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ load_P_each[5],_f,_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(3007,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ load_P_each[6],_f,_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(3008,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ load_P_each[7],_f,_f,_f,_f,_f,_f,_f])
	# change Q load
	psspy.load_chng_5(153,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[_f,load_Q_each[0],_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(154,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[_f, load_Q_each[1],_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(154,r"""2""",[_i,_i,_i,_i,_i,_i,_i],[_f, load_Q_each[2],_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(205,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[_f, load_Q_each[3],_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(203,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[_f, load_Q_each[4],_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(3005,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[_f,load_Q_each[5],_f,_f,_f,_f,_f,_f])
	psspy.load_chng_5(3007,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[_f,load_Q_each[6],_f,_f,_f,_f,_f,_f])
	# solve power flow
	psspy.fdns([0,0,0,1,1,3,99,0])
	print("***************Load_change_percentage: "+str(k)+"******************")
#psspy.save(r"""savnw_wind_scale_down.sav""")
# Further decrease P_gen and P_load at 206 and 205
psspy.machine_chng_2(206,r"""1""",[_i,_i,_i,_i,_i,_i],[ gen_P_each[2]-120,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
psspy.load_chng_5(205,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ load_P_each[4]-120,_f,_f,_f,_f,_f,_f,_f])

# Further decrease P_gen and P_load at 3008 and 3018

psspy.machine_chng_2(3018,r"""1""",[_i,_i,_i,_i,_i,_i],[ gen_P_each[5]-20,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])
psspy.load_chng_5(3008,r"""1""",[_i,_i,_i,_i,_i,_i,_i],[ load_P_each[7]-20,_f,_f,_f,_f,_f,_f,_f])

# run power flow
psspy.fdns([0,0,0,1,1,3,99,0])
psspy.save(r"""savnw_wind_scale_down.sav""")
