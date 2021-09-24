import sys, os, gc, random
from utils import *
PSSBIN_PATH = r'C:\Program Files (x86)\PTI\PSSEXplore34\PSSBIN'
PSSPY_PATH = r'C:\Program Files (x86)\PTI\PSSEXplore34\PSSPY34'
os.environ['PATH'] += ';' + PSSBIN_PATH
import psspy as ps
import redirect
import dyntools
redirect.psse2py()
import numpy as np
from collections import OrderedDict
#import pandas as pd
import csv
import win32com.client
import matplotlib.pyplot as plt
from solar_inverter import solar_inverter

_i = ps.getdefaultint()
_f = ps.getdefaultreal()
_s = ps.getdefaultchar()


# a class to store parameters for PV generators
class pv_model():
    def __init__(self, bus, vll, pg):
        # network information
        self.bus = bus
        self.pg = pg
        self.vll = vll
        self.vm = 0
        self.va = 0
        self.im = 0
        self.ia = 0

        # actual PV parameters
        self.get_params()

    def get_params(self):
        self.para = {}
        self.para['dt'] = 0.0001 #### simulation step size
        self.para['Sb'] = self.pg #### nominal power
        self.para['Vb'] = self.vll #### nominal voltage
        self.para['f'] = 60
        self.para['w_n'] = self.para['f']*2*np.pi
        self.para['L_f'] =1.35e-3
        self.para['C_f'] =50e-6
        self.para['r_f'] =0.1
        self.para['w_c'] = 31.41
        self.para['K_pv'] = 0.05
        self.para['K_iv'] = 390
        self.para['K_pc'] = 10.5
        self.para['K_ic'] = 16e3
        self.para['F'] = 0.75
        self.para['Tf'] = 1.5*self.para['Sb']
        self.para['Ta'] = 1
        self.para['Tv'] = 10
        self.para['Df'] = 1.5*self.para['Sb']
        self.para['Da'] = 0.2/self.para['Sb']
        self.para['Dv'] = 0.2/7*self.para['Vb']/self.para['Sb']

    # build the PV model using init. cond. and params
    def build(self):
        self.model = solar_inverter(self.im, self.ia, self.vm, self.va, para=self.para)

    # advance one global step
    def advance(self, dist, ts):
        step_amt = int(np.floor(ts / self.para['dt']))
        for i in range(step_amt):
            # update internal state
            self.model.cal_next_step()

            # plug in PQ
            net_P = self.model.time_varying_state['P']
            net_Q = self.model.time_varying_state['Q']
            dist.add_pv_net(self.bus, net_P[-1], net_Q[-1])
            dist.solve_ss()
            #print(net_P, net_Q)
            # get terminal state
            _, _, im, ia = dist.get_pv_terminal_states(self.bus)

            # next step
            self.model.cal_next_step_algebraic(im, ia)

            
# a class to store fault information
class trans_fault():
    def __init__(self, trans):
        self.trans = trans
        self.rand_type()
        self.rand_location()
        if 'fault' in self.type:
            self.rand_R()
        else:
            self.R = None
            self.X = None

    # the type of the fault (bus/gen/branch)
    def rand_type(self):
        self.type = np.random.choice(['gen_trip', 'bus_fault', 'bus_trip', 'branch_fault', 'branch_trip','FO'])
        #self.type = np.random.choice(['gen_trip'])
        
    # the location of the fault               
    def rand_location(self):
        if self.type == 'gen_trip':
            genid = np.random.choice(range(self.trans.gen_num))
            self.loc = (self.trans.gen_bus[genid], self.trans.gen_ids[genid])
            #self.loc = (3011, '1')
        elif self.type == 'bus_fault' or self.type == 'bus_trip':
            self.loc = np.random.choice(self.trans.bus_ids)
        elif self.type == 'branch_fault' or self.type == 'branch_trip':
            brnid = np.random.choice(range(self.trans.line_num))
            self.loc = [self.trans.bus_ids[i] for i in self.trans.line_T[brnid]]
        elif self.type == 'FO':
            genid = np.random.choice(range(self.trans.gen_num-1))
            self.loc = (self.trans.gen_bus[genid], self.trans.gen_ids[genid])
            self.dev1 = 0
            # random frequency based on location
            if self.trans.gen_bus[genid] == 101:
                self.freq = np.random.uniform(0.5,1.875)
            elif self.trans.gen_bus[genid] == 102:
                self.freq = np.random.uniform(0.5,1.775)
            elif self.trans.gen_bus[genid] == 206:
                self.freq = np.random.uniform(0.5,1.55)
            elif self.trans.gen_bus[genid] == 211:
                self.freq = np.random.uniform(0.7, 1.2)
            elif self.trans.gen_bus[genid] == 3011:
                self.freq = np.random.uniform(0.025,1.775)
            else:
                print('invalid FO source!')

        else:
            print('invalid fault type!')

    # random fault impedance
    def rand_R(self):
        # corresponding to low, med, high res fault
        fault_r_range = [[0.002,0.01],[0.01, 0.1],[0.1,1],[1,15]]
        fault_r = fault_r_range[np.random.choice([0,1,2,3])]
        #fault_r = fault_r_range[0]
        R = np.random.uniform(fault_r[0],fault_r[1])
        self.R = round(R, 4)
        self.X = R * np.random.uniform(0.2,3)

    # print out the fault
    def show(self):
        print(self.type, self.loc)

    # add this fault into system
    def apply(self, t=0):
        if self.type == 'gen_trip':
            ps.dist_machine_trip(self.loc[0], self.loc[1])
        elif self.type == 'bus_fault':
            ps.dist_bus_fault(self.loc, 3, 0.0, [self.R, self.X])
        elif self.type == 'bus_trip':
            ps.dist_bus_trip(self.loc)
        elif self.type == 'branch_fault':
            ps.dist_branch_fault(self.loc[0], self.loc[1], '1', 3, 0.0, [self.R, self.X])
        elif self.type == 'branch_trip':
            ps.dist_branch_trip(self.loc[0], self.loc[1], '1')
        elif self.type == 'FO':
            dev2 = 0.01*np.sin(2*3.14*self.freq*t)
            ps.increment_vref(self.loc[0],self.loc[1], dev2-self.dev1)
            self.dev1 = dev2
        else:
            print('invalid fault type!')

    # clear the fault
    def clear(self):
        ps.dist_clear_fault()

# a class for distribution system fault
class dist_fault():
    def __init__(self, buses, phases, ts, trans_bus):
        self.GNDFlag = 1
        self.trans_bus = trans_bus
        self.bus = self.rand_bus(buses[2:], phases[2:])
        self.phases = self.rand_phase(buses[2:], phases[2:])
        self.R = self.rand_resistance()
        self.T = self.rand_time(ts)
        self.cmd = self.get_cmd_string()

    # location of fault        
    def rand_bus(self, buses, phases):
        # return a random bus in the system
        self.bus_idx = np.random.choice(range(len(buses)))
        if self.GNDFlag:
            # 1 or 3-phase buses if GND path exists
            while not (len(phases[self.bus_idx])==1 or len(phases[self.bus_idx])==3):
                self.bus_idx = np.random.choice(range(len(buses)))
        else:
            # only 3-phase buses if GND path does not exist
            while not len(phases[self.bus_idx])==3:
                self.bus_idx = np.random.choice(range(len(buses)))            
        
        return buses[self.bus_idx]

    # return a fault type
    def rand_phase(self, buses, phases):
        p = phases[self.bus_idx]

        # if 1p line, only SLG possible 
        if len(p) == 1:
            self.type = '1'
            return str(p[0])

        # if 2p line, SLG, LL or LLG
        if len(p) == 2:
            if self.GNDFlag:
                self.type = np.random.choice(['1','2'])
            else:
                self.type = '2'
                
            if self.type == '1':
                return np.random.choice(p)
            elif self.type == '2' or self.type == '2g':
                return np.random.choice(p, 2, replace=False)
        
        # if 3p line, can have all kinds of fault
        elif len(p) == 3:
            if self.GNDFlag:
                self.type = np.random.choice(['1','2','3'])
            else:
                self.type = np.random.choice(['2','3'])
                
            if self.type == '1':
                return np.random.choice(['1','2','3'])
            elif self.type == '2' or self.type == '2g':
                return np.random.choice(['1','2','3'], 2, replace=False)
            else:
                return ['1','2','3']

        
    def rand_resistance(self):
        return 0.0001

    def rand_time(self, ts):
        return round(((np.floor(np.random.uniform(15, 30))+0.1) * ts), 4)
        #return round((4.1 * ts), 4)

    # generate DSS command string from randomized attributes
    def get_cmd_string(self):
        cmd = 'New Fault.F1 '
        # number of phases
        cmd += 'Phases=' + str(len(self.phases))
        # format the faulted lines to the input form
        if self.type == '1':
            cmd += ' Bus1=' + self.bus + '.' + self.phases[0]
        elif self.type == '2':
            cmd += ' Bus1=' + self.bus + '.' + self.phases[0] + '.0'
            cmd += ' Bus2=' + self.bus + '.' + self.phases[1] + '.0'
        elif self.type == '2g':
            cmd += ' Bus1=' + self.bus + '.' + self.phases[0] + '.' + self.phases[0]
            cmd += ' Bus2=' + self.bus + '.' + self.phases[1] + '.0'
        elif self.type == '3':
            cmd += ' Bus1=' + self.bus + '.1.2.3'
        # fault resistance
        cmd += ' R=' + str(self.R)
        # fault time
        cmd += ' ONtime=' + str(self.T)

        return cmd
    
    # apply the fault using a DSS handle
    def apply(self, txt):
        txt.Command = self.cmd
            

# class for trans system handler
class trans_handle():
    def __init__(self, path, ts):
        self.path = path
        self.ts = ts
        # load and parse the network info
        self.reset()
        self.load_case()
        self.get_network_info()

        
    def reset(self):
        with silence():
            # initialize PSSE
            ps.psseinit(50)
            ps.dynamics_solution_param_2(realar3=self.ts)
    
    def load_case(self):
        with silence():
            ps.read(0, self.path)

    def solve_ss(self):
        with silence():
            ps.fnsl([0,0,1,0,0,1,7,1])
        
    # process case and get network informations
    # create the following fields
    # bus_ids, bus_num, line_num, line_T, load_buses, load_ids, load_num
    def get_network_info(self):
        with silence():
            # list of bus names
            self.bus_ids = ps.abusint(-1, 2, 'NUMBER')[1][0]
            self.bus_num = ps.abuscount(-1,2)[1]

            # list of lines
            self.line_num = ps.abrncount(-1,1,3,4,1)[1]
            
            brnF = ps.abrnint(-1,1,3,4,1,'FROMNUMBER')[1][0]
            brnT = ps.abrnint(-1,1,3,4,1,'TONUMBER')[1][0]
            self.line_T = []

            # self.line_T stores the INDEX of buses in self.bus_ids, not the ID 
            for n in range(self.line_num):
                F = brnF[n]
                T = brnT[n]
                self.line_T.append((self.bus_ids.index(F),self.bus_ids.index(T)))
                
            self.line_ids = ps.abrnchar(-1, 1, 3, 4, 1, 'ID')[1][0]
            # list of load
            self.load_bus = ps.aloadint(-1, 4, 'NUMBER')[1][0]
            self.load_num = ps.aloadcount(-1, 4)[1]
            self.load_ids = ps.aloadchar(-1, 4, 'ID')[1][0]
            self.load_P = np.real(ps.aloadcplx(-1, 4, 'MVANOM')[1][0])
            self.load_Q = np.imag(ps.aloadcplx(-1, 4, 'MVANOM')[1][0])

            # generators
            self.gen_bus = ps.amachint(-1,4,'NUMBER')[1][0]
            self.gen_num = ps.amachcount(-1,1)[1]
            self.gen_ids = ps.amachchar(-1,1,'ID')[1][0]
            self.gen_cap = ps.amachreal(-1,1,'MBASE')[1][0]
            
    

    def change_load(self, bus, lid, P, Q):
        with silence():
            ps.load_chng_5(bus, lid, realar1=P, realar2=Q)


    def dyn_init(self):
        with silence():
            # convert load model
            #ps.conl(-1,1,1)
            #ps.conl(-1,1,2,[0,0],[100,0,0,100])
            #ps.conl(-1,1,3)
            # mach model
            ps.cong()
            ps.machine_data_2(3018,r"""1""",[_i,_i,_i,_i,_i,2],[_f,_f,_f,_f,_f,_f,_f,_f, 99999.0,_f,_f,_f,_f,_f,_f,_f,_f])

            # pre-processing
            ps.ordr()
            ps.fact()
            ps.tysl()

            # add case
            ps.dyre_new(dyrefile=self.path.replace('.raw','.dyr'))

            # add output channel
            ps.delete_all_plot_channels()
            ps.chsb(sid=0,all=1, status=[-1,-1,-1,1,13,0]) # bus volt
            # brn PQ
            for i in range(self.line_num):
                t = self.line_T[i]
                ps.branch_p_and_q_channel(status4=self.bus_ids[t[0]], \
                                          status5=self.bus_ids[t[1]],id=self.line_ids[i])

            # start dyn sim
            out_file = self.path.replace('.raw','.out')
            ps.strt_2(outfile=out_file) 


    # change the capacity of the wind machine at 3018
    def change_wind(self, pg_wind, pf = 1):
        with silence():
            ps.machine_chng_2(3018, '1', realar1 = pg_wind, realar2 = 0, \
                              realar3 = 0, realar4 = 0, realar5 = pg_wind, \
                              realar6 = pg_wind)

    # calculate the PG for all machines at PV buses
    def get_dispatch(self):
        with silence():
            # sum of all loads
            pd_actual = np.real(ps.aloadcplx(-1, 4, 'MVAACT')[1][0])
            pd_total = sum(pd_actual)

            # PG TBD, all load - wind gen
            wind_total = ps.macdat(3018, '1', 'P')[1]
            pg_total = pd_total - wind_total

            # split pg_total accross generators
            trad_gen_bus = [101,102,206,211,3011]
            pct_all = sum([9,9,10,7.25,10])
            trad_gen_pct = [9,9,10,7.25,10] 

            # modify PG of generators
            for i in range(5):
                ps.machine_chng_2(trad_gen_bus[i], '1', realar1=pg_total * \
                                  trad_gen_pct[i] / pct_all)
            
    # close process
    def close(self):
        ps.pssehalt_2()
            
# class for dist system handler
class dist_handle():
    def __init__(self):

        # initialize DSS interface objects
        self.dss_handle = win32com.client.Dispatch("OpenDSSEngine.DSS")
        self.txt = self.dss_handle.Text
        self.ckt = self.dss_handle.ActiveCircuit
        self.sol = self.ckt.Solution
        self.ActElmt = self.ckt.ActiveCktElement
        self.ActBus = self.ckt.ActiveBus 


    def new_ckt(self, path, ts):
        self.path = path
        self.ts = ts

        # load and parse the network info
        self.reset()
        self.load_case()
        self.get_network_info()
        self.txt.Command = "Set maxcontroliter=50"
        self.txt.Command = "Set maxiterations=50"

    def reset(self):
        self.dss_handle.ClearAll()
        self.sol.MaxIterations = 50
        self.sol.MaxControlIterations = 50

        
    def load_case(self):
        self.txt.Command = "compile [{}]".format(self.path)

    def solve_ss(self):
        self.sol.Solve()

    # process case and get network informations
    # create the following fields
    # bus_names, bus_num, bus_phases, line_names, line_num, line_T,
    # xfmr_T, xfmr_names, xfmr_num
    def get_network_info(self):
        # list of bus names
        self.bus_names = self.ckt.AllBusNames
        self.bus_num = len(self.bus_names)
        # list of phases for each bus
        self.bus_phases = []
        for n in self.bus_names:
            self.ckt.SetActiveBus(n)
            self.bus_phases.append(self.ActBus.Nodes)


        # list of lines
        self.line_names = self.ckt.Lines.AllNames
        self.line_num = self.ckt.Lines.Count

        self.line_T = []
        for n in self.line_names:
            full_name = 'line.' + n
            self.ckt.SetActiveElement(full_name)
            F = self.ActElmt.Properties('Bus1').val.split('.')[0]
            T = self.ActElmt.Properties('Bus2').val.split('.')[0]
            
            # take only the 3-phase bus name 
            self.line_T.append((self.bus_names.index(F),self.bus_names.index(T)))

        # add transformers as lines (for graph making purpose)
        self.xfmr_names = self.ckt.Transformers.AllNames
        self.xfmr_num = self.ckt.Transformers.Count
        self.xfmr_T = []
        for tr in self.xfmr_names:
            full_name = 'Transformer.' + tr
            self.ckt.SetActiveElement(full_name)
            F = self.bus_names.index(self.ActElmt.busNames[0].split('.')[0])
            T = self.bus_names.index(self.ActElmt.busNames[1].split('.')[0])

            self.xfmr_T.append((F,T))
        self.xfmr_T = list(self.xfmr_T)


    # get line or xfmr current measurement using line name
    def get_line_PQ(self, name, field):
        full_name = field + '.' + name
        self.ckt.SetActiveElement(full_name)
        res = self.ActElmt.SeqPowers
        assert len(res) == 12, "This line has less than 3 conductors!"
            
        return res[0:6]

    #
    def get_source_PQ(self):
        source_PQ_seq = np.zeros(6)
        # go through lines
        for l in range(self.line_num):
            orig = self.line_T[l][0]
            if orig == 0: # this line orginates from source
                res = self.get_line_PQ(self.line_names[l],'line')
                source_PQ_seq += res
        # transformers
        for t in range(self.xfmr_num):
            orig = self.xfmr_T[t][0]
            if orig == 0:
                res = self.get_line_PQ(self.xfmr_names[t], 'transformer')
                source_PQ_seq += res
        
        # total P/Q for this feeder, assume balanced
        tot_P = source_PQ_seq[0] + source_PQ_seq[2] + source_PQ_seq[4]
        tot_Q = source_PQ_seq[1] + source_PQ_seq[3] + source_PQ_seq[5]

        return tot_P / 1000, tot_Q / 1000

    # set source voltage of a feeder
    def set_source_V(self, V):
        self.ckt.Vsources.First
        self.ckt.Vsources.pu = V

    # scale all loads in the system
    def scale_loads(self, pct):
        loads = self.ckt.Loads
        loadNum = loads.Count
        # start from the first load
        loads.First
        for i in range(loadNum):
            loadP = loads.kW
            loadQ = loads.kvar
            loads.kW = loadP * pct
            loads.kvar = loadQ * pct
            loads.Next

    # add a new equivalent load for the PV gens
    def add_pv_load(self, bus):
        self.txt.Command = "new Load.pv"+str(bus)+" Bus1="+str(bus)+".1.2.3 Phases=3 Conn=Delta Model=1 kV=34.5 kW=0 kvar=0"

    # set the net PG for the PV generator
    def add_pv_net(self, bus, pg, qg=0):
        name = "pv" + str(bus)
        self.ckt.Loads.Name = name
        self.ckt.Loads.kW = -pg / 1000
        if qg > 100:
            self.ckt.Loads.kvar = -qg / 1000

    def get_pv_terminal_states(self, bus):
        name = "Load.pv" + str(bus)
        self.ckt.SetActiveElement(name)
        vma = self.ActElmt.VoltagesMagAng
        ima = self.ActElmt.CurrentsMagAng
        
        vm = (vma[0] + vma[2] + vma[4]) / 3
        va = (vma[1] + vma[3] + vma[5]) * 2 * 3.14159 / 540
        im = (ima[0] + ima[2] + ima[4])
        ia = (ima[1] + ima[3] + ima[5]) * 2 * 3.14159 / 540 - 1.04719

        return vm, va, im, ia

    # initialize dynamic simulation
    def dyn_init(self):
        self.txt.Command = "Set Maxcontroliter=50"
        self.txt.Command = "Set maxiterations=50"
        self.txt.Command = "solve"
        self.txt.Command = "set mode=dynamics stepsize=".format(self.ts)

# the aggregated input files
class simdata():
    def __init__(self, path, num=-1, version=1):
        ## load wind/pv/load data from input (v1)
        # num: -1 -> all, >-1 -> number of rows to be read
        self.ts = []
        self.wind_ts = []
        self.pv_ts = []
        self.load_ts = []
        if version == 1:

            line = 0
            with open(path) as fh:
                csv_file = csv.reader(fh)
                for row in csv_file:
                    if line > 0:
                        try:
                            self.ts.append(row[0])
                            self.wind_ts.append([float(row[i]) for i in [1,2,3]])
                            self.pv_ts.append([float(row[i]) for i in [4,5,6]])
                            self.load_ts.append([float(row[i]) for i in [7,8,9,10,11,12,13]])
                        except:
                            pass
                    line += 1
                    if line == num:
                        break
        # load v3 data
        # 7 loads, 2 pv, 1 wind
        elif version == 3:
            line = 0
            with open(path) as fh:
                csv_file = csv.reader(fh)
                for row in csv_file:
                    if line > 0:
                        try:
                            self.ts.append(row[0])
                            self.wind_ts.append([float(row[i]) for i in [10]])
                            self.pv_ts.append([float(row[i]) for i in [8,9]])
                            self.load_ts.append([float(row[i]) for i in [1,2,3,4,5,6,7]])
                        except:
                            pass
                    line += 1
                    if line == num:
                        break        

class cosim():

    # T_path -- RAW file of a PSSE trans case
    # D_paths -- list of a bunch of DSS files
    # D_Bus -- Bus ID for each dist case in D_path
    def __init__(self, T_path, D_paths, D_bus, data):
        self.T_path = T_path
        self.D_paths = D_paths
        self.D_num = len(self.D_paths)
        self.D_bus = D_bus
        self.D_bus_unique = list(set(self.D_bus))
        self.D_num_unique = len(self.D_bus_unique)
        
        self.dyn_ts = 1/240

        # load profile data
        self.ts = data.ts
        self.wind_ts = data.wind_ts
        self.pv_ts = data.pv_ts
        self.load_ts = data.load_ts

        # initialize dist sys
        self.dist = dist_handle()

    # load transmission system
    def load_trans(self):
        # load transmission case
        self.trans = trans_handle(self.T_path, self.dyn_ts)

    ## solve SS power flow iteratively
    # row_idx: corresponding row number for Profiles
    def solve_ss(self, row_idx):
        self.load_trans()
        
        # loop until converged
        eta = 1e-3 # convergence threshold
        done = 0
        iternum = 0
        
        # flat voltage for all D nodes
        D_volts = np.ones(self.D_num)

        # load wind profile and set gen 3018 (1.0 PF)
        pg_wind = self.wind_ts[row_idx][0] * 130
        self.trans.change_wind(pg_wind)

        # load PV models, one PV each feeder
        self.pvs = []
        # pv init. cond.
        pvs_q = np.zeros(self.D_num)
        for i in range(self.D_num):
            pg_pv = self.pv_ts[row_idx][i] * 34500
            self.pvs.append(pv_model(671, pg_pv, 5000000))
        
        # set bus load in trans for all non-feeder buses
        for i in range(self.trans.load_num):
            if self.trans.load_bus[i] in self.D_bus_unique:
                pass
            else:
                # modify buses without distribution system
                P_new = self.trans.load_P[i] * self.load_ts[row_idx][i]
                Q_new = self.trans.load_Q[i] * self.load_ts[row_idx][i]

                self.trans.change_load(self.trans.load_bus[i], '1', \
                            P_new, Q_new)

        while not done:
            iternum += 1
            # total PQ for bus in T with D model
            D_P_unique = np.zeros(self.D_num_unique)
            D_Q_unique = np.zeros(self.D_num_unique)
            ## start from D, loop through all D system
            for d_idx in range(self.D_num):
                self.dist.new_ckt(self.D_paths[d_idx], self.dyn_ts)
                
                # load profiles and DER output
                load_ts_idx = self.trans.load_bus.index(self.D_bus[d_idx])
                self.dist.scale_loads(self.load_ts[row_idx][load_ts_idx])

                # load PV net output
                pv_bus = self.pvs[d_idx].bus
                pv_pg = self.pvs[d_idx].pg
                # add negative load for PV
                self.dist.add_pv_load(pv_bus)
                self.dist.add_pv_net(pv_bus, pv_pg)
                                    
                # set voltage
                self.dist.set_source_V(D_volts[d_idx])

                # solve D system, get sub power draw
                self.dist.solve_ss()
                assert self.dist.sol.Converged, "DSS PF Failed!"

                # get PV initial measurements
                vm, va, im, ia = self.dist.get_pv_terminal_states(self.pvs[d_idx].bus)
                self.pvs[d_idx].vm = vm
                self.pvs[d_idx].va = va
                self.pvs[d_idx].im = im
                self.pvs[d_idx].ia = ia
                
                # get all PQ leaving the sourcebus for this D system
                tot_P, tot_Q = self.dist.get_source_PQ()

                # add the PQ to the bus in D
                D_idx_unique = self.D_bus_unique.index(self.D_bus[d_idx])
                D_P_unique[D_idx_unique] += tot_P
                D_Q_unique[D_idx_unique] += tot_Q

                
                self.dist.reset()

            # add fixed load to 3008 (TEMP)
            D_P_unique[0] += self.load_ts[row_idx][6] * 95
            D_Q_unique[0] += self.load_ts[row_idx][6] * 60
            
            # set bus load in trans for all feeder buses
            for D_idx_unique in range(self.D_num_unique):
                self.trans.change_load(self.D_bus_unique[D_idx_unique], '1', \
                                       D_P_unique[D_idx_unique], D_Q_unique[D_idx_unique])


            # change PG according to all loads
            self.trans.get_dispatch()
            #print(sum(np.real(ps.aloadcplx(-1, 4, 'MVAACT')[1][0])))
            #print(sum(ps.amachreal(-1,1,'PGEN')[1][0]))
            
            # solve trans
            self.trans.solve_ss()
            
            # new voltages magnitude for dist buses
            prev_volts = D_volts.copy()
            all_volts = ps.abusreal(-1, 2, 'PU')[1][0]
            D_volts = np.array([all_volts[self.trans.bus_ids.index(i)] for i in self.D_bus])

            ## check for convergence
            if np.linalg.norm(D_volts-prev_volts) < eta or iternum > 20:
                done = 1
                #print('SSPF converged after {} iterations'.format(iternum))



    # create a random fault scenario
    # output: self.fault
    def create_rand_dist(self):
        if random.uniform(0, 1) > 0.95:
            trans_bus = np.random.choice(self.D_bus)
            self.dist.new_ckt(self.D_paths[0], self.dyn_ts)
            self.fault = dist_fault(self.dist.bus_names, self.dist.bus_phases, \
                                    self.dyn_ts, trans_bus)
            self.fault_type = 'dist'
        else:
            self.fault = trans_fault(self.trans)
            self.fault_type = 'trans'


    # perform dynamic simulation
    def dynsim(self, row_idx, max_steps):

        # load profiles and solve init condition
        self.solve_ss(row_idx)

        # initialize PSSE for dyn sim
        self.trans.dyn_init()

        # create a random fault scenario
        self.create_rand_dist()
        #self.fault.show()
        fault_step = np.random.choice(range(10, int(np.floor(max_steps/5))))
        clear_step = np.random.choice(range(int(np.floor(max_steps*2/3)), int(np.floor(max_steps*4/3))))
        #fault_step = 159
        #clear_step = 9999
        #print(fault_step, clear_step)

        # store the fault information
        fault_info = OrderedDict()
        fault_info['datetime'] = self.ts[row_idx]
        fault_info['class'] = self.fault_type
        fault_info['start'] = round(fault_step*self.dyn_ts,4)
        if clear_step >= max_steps:
            fault_info['end'] = -1
        else:
            fault_info['end'] = round(clear_step*self.dyn_ts,4)

        # trans fault specific data
        if self.fault_type == 'trans':
            if 'branch' in self.fault.type:
                fault_info['bus1'] = self.fault.loc[0]
                fault_info['bus2'] = self.fault.loc[1]
            elif 'gen' in self.fault.type:
                fault_info['bus1'] = self.fault.loc[0]
                fault_info['bus2'] = -1
            elif 'FO' == self.fault.type:
                fault_info['bus1'] = self.fault.loc[0]
                fault_info['bus2'] = -1
                fault_info['frequency'] = self.fault.freq
            else:
                fault_info['bus1'] = self.fault.loc
                fault_info['bus2'] = -1
            fault_info['type'] = self.fault.type
        # dist fault specific data
        elif self.fault_type == 'dist':
            fault_info['bus1'] = str(self.fault.trans_bus) + '.' + str(self.fault.bus)
            fault_info['bus2'] = -1
            fault_info['type'] = self.fault.phases
        
        # initial dist V array
        all_volts = ps.abusreal(-1, 2, 'PU')[1][0]
        #print(all_volts)
        D_volts = np.zeros((max_steps, self.D_num_unique))
        D_volts[0,:] = [all_volts[self.trans.bus_ids.index(i)] for i in self.D_bus_unique]

        # initialize PV models
        for d_idx in range(self.D_num):
            self.pvs[d_idx].build()

        # initialize containers
        times = []
        trans_ts = OrderedDict()
        dist_ts = OrderedDict()

        # dyn loop

        for t in range(max_steps):
            # loop through all D s
            # total PQ for bus in T with D model
            D_P_unique = np.zeros(self.D_num_unique)
            D_Q_unique = np.zeros(self.D_num_unique)
            # create containers
            if t == 0:
                dist_ts['Time(s)'] = []
            dist_ts['Time(s)'].append(round(t*self.dyn_ts,4))
            for d_idx in range(self.D_num):
                
                # initialize DSS
                self.dist.new_ckt(self.D_paths[d_idx], self.dyn_ts)
                
                # load initial V, for the common load bus
                if t > 0:
                    self.dist.set_source_V(D_volts[t-1, self.D_bus_unique.index(self.D_bus[d_idx])])
                else:
                    self.dist.set_source_V(D_volts[t, self.D_bus_unique.index(self.D_bus[d_idx])])
                
                # load SS demand data (WIP)
                load_ts_idx = self.trans.load_bus.index(self.D_bus[d_idx])
                self.dist.scale_loads(self.load_ts[row_idx][load_ts_idx])

                # add fault
                if self.fault_type == 'dist':
                    if t >= fault_step and self.fault.trans_bus == self.D_bus[d_idx]:
                        self.fault.apply(self.dist.txt) 
                
                # get PV net PQ (WIP)
                pv_bus = self.pvs[d_idx].bus
                pv_pg = self.pvs[d_idx].pg
                self.dist.add_pv_load(pv_bus)
                self.pvs[d_idx].advance(self.dist, self.dyn_ts)
                
                # solve D, get sub power draw
                self.dist.solve_ss()
                assert self.dist.sol.Converged, "DSS PF Failed!"

                # get all PQ leaving the sourcebus for this D system
                tot_P, tot_Q = self.dist.get_source_PQ()

                # add the PQ to the bus in D
                D_idx_unique = self.D_bus_unique.index(self.D_bus[d_idx])
                D_P_unique[D_idx_unique] += tot_P
                D_Q_unique[D_idx_unique] += tot_Q

                # store dist voltages to container
                # create container
                node_ct = len(self.dist.ckt.AllNodeNames)

                if t == 0:
                    for n in range(node_ct):
                        dist_ts[str(self.D_bus[d_idx])+'.'+self.dist.ckt.AllNodeNames[n]] = []

                for n in range(node_ct):
                    dist_ts[str(self.D_bus[d_idx])+'.'+self.dist.ckt.AllNodeNames[n]].append(round(self.dist.ckt.AllBusVmagPu[n], 4))
                

                

            ## DEBUG
            #print(D_P_unique, D_Q_unique)
            
            # add fixed load to 3008 (TEMP)
            D_P_unique[0] += self.load_ts[row_idx][6] * 95
            D_Q_unique[0] += self.load_ts[row_idx][6] * 60

            # set bus load in trans for all feeder buses
            for D_idx_unique in range(self.D_num_unique):
                self.trans.change_load(self.D_bus_unique[D_idx_unique], '1', \
                                       D_P_unique[D_idx_unique], D_Q_unique[D_idx_unique])


            # run dynamic PF
            with silence():
                if self.fault_type == 'trans':
                    # add disturbances
                    if t == fault_step:
                        self.fault.apply()
                    # clear disturbances
                    if t == clear_step:
                        self.fault.clear()
                    if self.fault.type == 'FO' and t >= fault_step and t < clear_step:
                        self.fault.apply(t*self.dyn_ts)
                        

                # solve trans step
                ps.run(tpause=t*self.dyn_ts)
            
            # get dist voltage
            fobj = dyntools.CHNF(self.trans.path.replace('.raw','.out'))
            
            # read and store dyn output data
            short_title, cid, cdata = fobj.get_data()
            # initialize containers in the 1st step
            if t == 0:
                for key in cid:
                    trans_ts[cid[key]] = []

            # store measurements
            for key in cid:
                trans_ts[cid[key]].append(round(cdata[key][-1],4))

            # change distribution bus voltages
            for D_idx_unique in range(self.D_num_unique):
                D_bus_ind = self.trans.bus_ids.index(self.D_bus_unique[D_idx_unique])
                D_volts[t,D_idx_unique] = cdata[D_bus_ind][-1]
            times.append(t*self.dyn_ts)

##            if t % 100 == 0:
##                print(t)
            #input()
            # DEBUG
            #self.dist.ckt.SetActiveBus('632')
            #print(self.dist.ckt.AllBusVmag)

        # plot dist terminal voltage
        plt.figure()
        plt.plot(times, D_volts[:,0])
        plt.show()

        self.trans.close()
        self.dist.reset()

        return trans_ts, dist_ts, fault_info
