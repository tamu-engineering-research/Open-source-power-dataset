# This file aims to generate an save file with wind generation;
# The result file is r"""savnw_wind.sav"""
import os
import pssexplore34
import psspy
import redirect
import dyntools
import numpy as np
import pssplot
psspy.psseinit()
#redirect.py2psse()
from psspy import _i, _f, _s, _o


# SET FILE names
study   = 'savnw_REwind'
suffix  = '_flat'
savfile = r"""savnw.sav"""
conlfile= r"""savnw_Conl.idv"""
cnvfile = '%s_cnv.sav'%study
dyrfile = '%s.dyr'%study
snpfile = '%s.snp'%study
outfile = '%s%s.out'%(study,suffix)
logfile = '%s%s.log'%(study,suffix)
psspy.progress_output(2,logfile,[0,0])
# -------------------------------------------------------------------------
# 1: LOAD PSSE CASE
#    'replace' gen at bus 3018 with new solar PV plant
# -------------------------------------------------------------------------
psspy.case(savfile)
psspy.solution_parameters_3([_i,100,_i],[_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f,_f])

# Convert Gen at 3018 to "Solar" [Xs=99999 and Wind Control Mode WCM =2 ]
psspy.machine_data_2(3018,r"""1""",[_i,_i,_i,_i,_i,2],[_f,_f,_f,_f,_f,_f,_f,_f, 99999.0,_f,_f,_f,_f,_f,_f,_f,_f])

# SAVE THE PSSE CASE
psspy.fdns([1,0,1,1,1,0,99,0])
psspy.save(r"""savnw_wind.sav""")

# -------------------------------------------------------------------------
# 2: convert case and create snp file
#    
# -------------------------------------------------------------------------