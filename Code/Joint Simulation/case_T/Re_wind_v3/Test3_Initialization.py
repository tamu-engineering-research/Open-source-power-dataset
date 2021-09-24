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

def Initialization(output_name):
	# first attempt to initialization
	psspy.strt_2([0,0],output_name) # initial check is not okay
	psspy.run(0, 10.0,0,1,0) # run a 10-second simulation
	psspy.strt_2([0,0],output_name) # initialization check is okay
	


sav_name = r"""C:\Users\Tong Huang\Desktop\FOL\WECC_ORIGINAL\Re_wind_v3\savnw_wind_scale_down.sav"""
dyn_name = r"""C:\Users\Tong Huang\Desktop\FOL\WECC_ORIGINAL\Re_wind_v3\savnw_REwind.dyr"""
output_name = r"""output1"""
psspy.case(sav_name)
psspy.fdns([0,0,0,1,1,3,99,0])
psspy.cong(0)
psspy.conl(0,1,1,[0,0],[ 100.0,0.0,0.0, 100.0])
psspy.conl(0,1,2,[0,0],[ 100.0,0.0,0.0, 100.0])
psspy.conl(0,1,3,[0,0],[ 100.0,0.0,0.0, 100.0])
psspy.save(r"""C:\Users\Tong Huang\Desktop\FOL\WECC_ORIGINAL\Re_wind_v3\savnw_wind_scale_down_cnv.sav""")
psspy.dyre_new([1,1,1,1],dyn_name,"","","")
# add channel
psspy.chsb(0,1,[-1,-1,-1,1,14,0])
psspy.snap([386,170,103,70,46],r"""C:\Users\Tong Huang\Desktop\FOL\WECC_ORIGINAL\Re_wind_v3\savnw_REwind.snp""")
# initialization
Initialization(output_name)