import numpy as np

def Dynamics_PowSensor(p_t, q_t, P0, Q0, w_c):
    """
    Power sensor dynamics
    pt, qt: instantanous power
    P0, Q0: last step P and Q
    """
    dP = -w_c*P0 + w_c*p_t
    dQ = -w_c*Q0 + w_c*q_t
    return dP, dQ

def Dynamics_LC_Filter(para_LC, i_ld0, i_lq0, v_od0, v_oq0, v_id0, v_iq0, i_od0, i_oq0, w0):
    """
    The function gives one-step update of the lc filter
    para_LC includes the parameters of the LC filter, e.g., para_LC.r_f
    the variables ending with 0 are the current step value
    i_ld, i_lq, v_od, v_oq, i_od, i_oq are next-step update
    dt: time step
    """
    r_f = para_LC['r_f']
    L_f = para_LC['L_f']
    C_f = para_LC['C_f']
    di_ld = -r_f/L_f*i_ld0 + w0 * i_lq0 + 1/L_f*(v_id0 - v_od0)
    di_lq = -r_f/L_f*i_lq0 - w0 * i_ld0 + 1/L_f*(v_iq0 - v_oq0)
    dv_od = w0*v_oq0 + 1/C_f*(i_ld0 -i_od0)
    dv_oq = -w0*v_od0 + 1/C_f*(i_lq0 -i_oq0)
    return di_ld, di_lq, dv_od, dv_oq

def Dyn_Freq_Droop(w0, v_od_star0, P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star):
    """
    This function defines the dynamics of the frequency droop controller
    w0, and v_od_star0 are the last step state variables
    """
    ddelta = w0 - w_n
    dw = 1/Tf*(-Df*(w0 - w_n) + P_star - P0)
    dv_od_star = 1/Tv*(Dv*(Q_star - Q0) - v_od_star0 + V_star)
    return ddelta, dw, dv_od_star

def dq2DQ(delta, d, q):
    D_trans = np.cos(delta)*d - np.sin(delta)*q
    Q_trans = np.sin(delta)*d + np.cos(delta)*q
    return D_trans, Q_trans

def DQ2dq(delta, D, Q):
    d_trans = np.cos(delta)*D + np.sin(delta)*Q
    q_trans = -np.sin(delta)*D + np.cos(delta)*Q
    return d_trans, q_trans

class solar_inverter():
    def __init__(self, I_mag, I_ang, V_mag, V_ang, para=None):
        """
        time-varying variables: 
        delta, w, P, Q, phi_d, phi_q,i_ld_star, i_lq_star, 
        gamma_d, gamma_q, v_id, v_iq, i_ld, i_lq, v_od, v_oq, i_od, i_oq
        """
        self.time_varying_state_name = ['delta', 'w', 'P', 'Q', 'phi_d', 'phi_q', 'i_ld_star', 'i_lq_star', 
            'gamma_d', 'gamma_q', 'v_id', 'v_iq', 'i_ld', 'i_lq', 'v_od', 'v_oq', 'i_od', 'i_oq']
        self.update_para(para=para)
        self.initialize_state(I_mag, I_ang, V_mag, V_ang,)
        pass

    def update_setting(self):
        pass

    def update_para(self, para=None):
        if para==None:
            self.para = {}
            self.para['dt'] = 0.001
            self.para['Sb'] = 10e3
            self.para['Vb'] = 220*np.sqrt(3)
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
            print(r'Finish updating DEFAULT parameters of LC filters and controllers \n')
        else:
            self.para = para
            #print(r'Finish updating INPUTTED parameters of LC filters and controllers \n')

    def initialize_state(self, I_mag, I_ang, V_mag, V_ang,):
        # update all states based on I_mag, I_ang, V_mag, V_ang at terminal in the solved power flow solution
        # self.time_varying_state = {}
        # for state_name in self.time_varying_state_name:
        #     self.time_varying_state[state_name] = []
        I_D = I_mag*np.cos(I_ang)
        I_Q = I_mag*np.sin(I_ang)
        V_D = V_mag*np.cos(V_ang)
        V_Q = V_mag*np.sin(V_ang)
        # power controller
        delta0 = V_ang ### make sure v_oq0=0 ###
        i_od0, i_oq0 = DQ2dq(delta0, I_D, I_Q)
        v_od0, v_oq0 = DQ2dq(delta0, V_D, V_Q)
        P0 = v_od0*i_od0 + v_oq0*i_oq0
        Q0 = -v_od0*i_oq0 + v_oq0*i_od0
        self.P_star = P0
        self.Q_star = Q0
        self.V_star = v_od0
        w0 = self.para['w_n']
        # LC filter
        i_ld0 = -self.para['w_n']*self.para['C_f']*v_oq0 + i_od0
        i_lq0 = self.para['w_n']*self.para['C_f']*v_od0 + i_oq0
        v_id0 = self.para['r_f']*i_ld0 - self.para['w_n']*self.para['L_f']*i_lq0 +v_od0
        v_iq0 = self.para['w_n']*self.para['L_f']*i_ld0 + self.para['r_f']*i_lq0 +v_oq0
        # current controller
        gamma_d0 = 1/self.para['K_ic']*(v_id0 + self.para['w_n']*self.para['L_f']*i_lq0)
        gamma_q0 = 1/self.para['K_ic']*(v_iq0 - self.para['w_n']*self.para['L_f']*i_ld0)
        # voltage controller
        phi_d0 = 1/self.para['K_iv']*(i_ld0 - self.para['F']*i_od0 + self.para['w_n']*self.para['C_f']*v_oq0)
        phi_q0 = 1/self.para['K_iv']*(i_lq0 - self.para['F']*i_oq0 - self.para['w_n']*self.para['C_f']*v_od0)
        # initializa time varying state list
        self.time_varying_state = {}
        self.time_varying_state['delta'] = [delta0]
        self.time_varying_state['i_od'] = [i_od0]
        self.time_varying_state['i_oq'] = [i_oq0]
        self.time_varying_state['v_od'] = [v_od0]
        self.time_varying_state['v_oq'] = [v_oq0]
        self.time_varying_state['v_od_star'] = [v_od0]
        self.time_varying_state['v_oq_star'] = [v_oq0]
        self.time_varying_state['P'] = [P0]
        self.time_varying_state['Q'] = [Q0]
        self.time_varying_state['w'] = [w0]
        self.time_varying_state['i_ld'] = [i_ld0]
        self.time_varying_state['i_lq'] = [i_lq0]
        self.time_varying_state['i_ld_star'] = [i_ld0]
        self.time_varying_state['i_lq_star'] = [i_lq0]
        self.time_varying_state['v_id'] = [v_id0]
        self.time_varying_state['v_iq'] = [v_iq0]
        self.time_varying_state['gamma_d'] = [gamma_d0]
        self.time_varying_state['gamma_q'] = [gamma_q0]
        self.time_varying_state['phi_d'] = [phi_d0]
        self.time_varying_state['phi_q'] = [phi_q0]
    
    def update_state(self):
        pass

    def cal_next_step(self,):
        ## get constant
        dt = self.para['dt']
        w_n = self.para['w_n']
        P_star = self.P_star
        Q_star = self.Q_star
        V_star = self.V_star
        ## get curremt state
        P0, Q0 = self.time_varying_state['P'][-1], self.time_varying_state['Q'][-1]
        v_od0, v_oq0 = self.time_varying_state['v_od'][-1], self.time_varying_state['v_oq'][-1]
        i_od0, i_oq0 = self.time_varying_state['i_od'][-1], self.time_varying_state['i_oq'][-1]
        v_id0, v_iq0 = self.time_varying_state['v_id'][-1], self.time_varying_state['v_iq'][-1]
        i_ld0, i_lq0 = self.time_varying_state['i_ld'][-1], self.time_varying_state['i_lq'][-1]
        gamma_d0, gamma_q0 = self.time_varying_state['gamma_d'][-1], self.time_varying_state['gamma_q'][-1]
        v_od_star0, v_oq_star0 = self.time_varying_state['v_od_star'][-1], self.time_varying_state['v_oq_star'][-1]
        i_ld_star0, i_lq_star0 = self.time_varying_state['i_ld_star'][-1], self.time_varying_state['i_lq_star'][-1]
        delta0, w0 = self.time_varying_state['delta'][-1], self.time_varying_state['w'][-1]
        phi_d0, phi_q0 = self.time_varying_state['phi_d'][-1], self.time_varying_state['phi_q'][-1]
        ## updating internal time varying variables of inverters defined by ODE
        delta, v_od_star, v_oq_star, w, p, q = self.power_controller(v_od0, v_oq0, i_od0, i_oq0, P0, Q0, delta0, v_od_star0, w0, w_n, P_star, Q_star, V_star, dt)
        phi_d, phi_q = self.voltage_controller(v_od_star0, v_oq_star0, v_od0, v_oq0, dt, phi_d0, phi_q0)
        gamma_d, gamma_q = self.current_controller(i_ld_star0, i_lq_star0, i_ld0, i_lq0, dt, gamma_d0, gamma_q0)
        i_ld, i_lq, v_od, v_oq = self.LC_filter(i_ld0, i_lq0, v_od0, v_oq0, v_id0, v_iq0, i_od0, i_oq0, w0, dt)
        ## append internal time varying variables into inverters
        self.time_varying_state['delta'].append(delta)
        self.time_varying_state['w'].append(w)
        self.time_varying_state['phi_d'].append(phi_d)
        self.time_varying_state['phi_q'].append(phi_q)
        self.time_varying_state['gamma_d'].append(gamma_d)
        self.time_varying_state['gamma_q'].append(gamma_q)
        self.time_varying_state['i_ld'].append(i_ld)
        self.time_varying_state['i_lq'].append(i_lq)
        self.time_varying_state['v_od'].append(v_od)
        self.time_varying_state['v_oq'].append(v_oq)
        self.time_varying_state['P'].append(p)
        self.time_varying_state['Q'].append(q)
        self.time_varying_state['v_od_star'].append(v_od_star)
        self.time_varying_state['v_oq_star'].append(v_oq_star)
    
    def get_latest_terminal_voltage(self):
        # help the external system to get the latest terminal voltage
        v_od = self.time_varying_state['v_od'][-1]
        v_oq = self.time_varying_state['v_oq'][-1]
        delta = self.time_varying_state['delta'][-1]
        v_oD, v_oQ = dq2DQ(delta, v_od, v_oq)
        V_mag = np.sqrt(v_oD**2+v_oQ**2)
        V_ang = np.angle(v_oD + 1j *v_oQ)
        return V_mag, V_ang

    def cal_next_step_algebraic(self, I_mag, I_ang):
        ## updating algebraic variables
        # terminal current dq 
        I_D = I_mag*np.cos(I_ang)
        I_Q = I_mag*np.sin(I_ang)
        delta = self.time_varying_state['delta'][-1]
        i_od0, i_oq0 = DQ2dq(delta, I_D, I_Q)
        self.time_varying_state['i_od'].append(i_od0)
        self.time_varying_state['i_oq'].append(i_oq0)
        # voltage controller setting
        i_ld_star, i_lq_star = self.VolCtr_alg()
        self.time_varying_state['i_ld_star'].append(i_ld_star)
        self.time_varying_state['i_lq_star'].append(i_lq_star)
        # current controller setting
        v_id, v_iq = self.CurCtr_alg()
        self.time_varying_state['v_id'].append(v_id)
        self.time_varying_state['v_iq'].append(v_iq)

    def power_controller(self, v_od, v_oq, i_od, i_oq, P0, Q0, delta0, v_od_star0, w0, w_n, P_star, Q_star, V_star, dt):
        para_pc = self.para
        Tf = para_pc['Tf']
        Tv = para_pc['Tv']
        Df = para_pc['Df']
        Dv = para_pc['Dv']
        w_c = para_pc['w_c']
        # dynamics of power sensor
        p_t = v_od*i_od + v_oq*i_oq
        q_t = -v_od*i_oq + v_oq*i_od
        # update P, Q using RK4
        x0 = np.array([P0, Q0]).copy()
        dP, dQ = Dynamics_PowSensor(p_t, q_t, P0, Q0, w_c)
        k1 = dt*np.array([dP, dQ]).copy()
        x1 = x0 + 0.5*k1
        [dP,dQ] = Dynamics_PowSensor(p_t, q_t, x1[0], x1[1], w_c)
        k2 = dt*np.array([dP, dQ]).copy()
        x2 = x0 + 0.5*k2
        [dP,dQ] = Dynamics_PowSensor(p_t, q_t, x2[0], x2[1], w_c)
        k3 = dt*np.array([dP, dQ]).copy()
        x3 = x0 + k3
        [dP,dQ] = Dynamics_PowSensor(p_t, q_t, x3[0], x3[1], w_c)
        k4 = dt*np.array([dP, dQ]).copy()
        x = x0 + 1/6*(k1 + 2*k2 + 2*k3 +k4)
        p = x[0]
        q = x[1]
        # update delta, w, v_od_star using RK4
        x0 = np.array([delta0, w0, v_od_star0]).copy()
        ddelta, dw, dv_od_star = Dyn_Freq_Droop(w0, v_od_star0, P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star)
        k1 = dt*np.array([ddelta, dw, dv_od_star]).copy()
        x1 = x0 + 0.5*k1
        [ddelta, dw, dv_od_star] = Dyn_Freq_Droop(x1[1], x1[2], P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star)
        k2 = dt*np.array([ddelta, dw, dv_od_star]).copy()
        x2 = x0 + 0.5*k2
        [ddelta, dw, dv_od_star] = Dyn_Freq_Droop(x2[1], x2[2], P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star)
        k3 = dt*np.array([ddelta, dw, dv_od_star]).copy()
        x3 = x0 + k3
        [ddelta, dw, dv_od_star] = Dyn_Freq_Droop(x3[1], x3[2], P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star)
        k4 = dt*np.array([ddelta, dw, dv_od_star]).copy()
        x = x0 + 1/6*(k1 + 2*k2 + 2*k3 +k4)
        delta = x[0]
        w = x[1]
        v_od_star = x[2]
        v_oq_star = 0
        return delta, v_od_star, v_oq_star, w, p, q

    def voltage_controller(self, v_od_star,v_oq_star, v_od0, v_oq0, dt, phi_d0, phi_q0):
        """
        This function updates the state variables in the voltage controllors
        dt: time step
        phi_d0, phi_q0: state variables in the last step
        Euler approach is applied
        """
        dphi_d = v_od_star - v_od0
        dphi_q = v_oq_star - v_oq0
        phi_d = phi_d0 + dt* dphi_d
        phi_q = phi_q0 + dt* dphi_q
        return phi_d, phi_q

    def current_controller(self,i_ld_star,i_lq_star, i_ld, i_lq, dt, gamma_d0, gamma_q0):
        """
        This function updates the state variables in the current controllors
        dt: time step
        gamma_d0, gamma_q0: state variables in the last step
        Euler approach is applied
        """
        dgamma_d = i_ld_star - i_ld
        dgamma_q = i_lq_star - i_lq
        gamma_d = gamma_d0 + dgamma_d*dt
        gamma_q = gamma_q0 + dgamma_q*dt
        return gamma_d, gamma_q

    def LC_filter(self, i_ld0, i_lq0, v_od0, v_oq0, v_id0, v_iq0, i_od0, i_oq0, w0,dt):
        """
        The function models the dynamics of a LC filter
        the variables ending with 0 are the current step value
        i_ld, i_lq, v_od, v_oq, i_od, i_oq are next-step update
        dt: time step
        RK4 INTEGRATION
        """
        para_LC = self.para
        x0 = np.array([i_ld0, i_lq0, v_od0, v_oq0,]).copy()
        di_ld, di_lq, dv_od, dv_oq = Dynamics_LC_Filter(para_LC, i_ld0, i_lq0, v_od0, v_oq0, v_id0, v_iq0, i_od0, i_oq0, w0)
        k1 = dt * np.array([di_ld, di_lq, dv_od, dv_oq]).copy()
        x1 = x0 + 0.5*k1
        di_ld, di_lq, dv_od, dv_oq = Dynamics_LC_Filter(para_LC, x1[0], x1[1], x1[2], x1[3], v_id0, v_iq0, i_od0, i_oq0, w0)
        k2 = dt * np.array([di_ld, di_lq, dv_od, dv_oq]).copy()
        x2 = x0 + 0.5*k2
        di_ld, di_lq, dv_od, dv_oq = Dynamics_LC_Filter(para_LC, x2[0], x2[1], x2[2], x2[3], v_id0, v_iq0, i_od0, i_oq0, w0);
        k3 = dt * np.array([di_ld, di_lq, dv_od, dv_oq]).copy()
        x3 = x0 + k3
        di_ld, di_lq, dv_od, dv_oq = Dynamics_LC_Filter(para_LC, x3[0], x3[1], x3[2], x3[3], v_id0, v_iq0, i_od0, i_oq0, w0);
        k4 = dt * np.array([di_ld, di_lq, dv_od, dv_oq]).copy()
        x = x0 + 1/6*(k1 + 2*k2 + 2*k3 +k4)
        i_ld = x[0]
        i_lq = x[1]
        v_od = x[2]
        v_oq = x[3]
        return i_ld, i_lq, v_od, v_oq
    
    def VolCtr_alg(self):
        F = self.para['F']
        c_f = self.para['C_f']
        k_pv = self.para['K_pv']
        k_iv = self.para['K_iv']
        i_od = self.time_varying_state['i_od'][-1]
        i_oq = self.time_varying_state['i_oq'][-1]
        v_od = self.time_varying_state['v_od'][-1]
        v_oq = self.time_varying_state['v_oq'][-1]
        v_od_star = self.time_varying_state['v_od_star'][-1]
        v_oq_star = self.time_varying_state['v_oq_star'][-1]
        phi_d = self.time_varying_state['phi_d'][-1]
        phi_q = self.time_varying_state['phi_q'][-1]
        w_n = self.para['w_n']
        i_ld_star = F*i_od - w_n*c_f*v_oq + k_pv*(v_od_star-v_od) + k_iv*phi_d
        i_lq_star = F*i_oq + w_n*c_f*v_od + k_pv*(v_oq_star-v_oq) + k_iv*phi_q
        return i_ld_star, i_lq_star
    
    def CurCtr_alg(self):
        k_pc = self.para['K_pc']
        k_ic = self.para['K_ic']
        l_f = self.para['L_f']
        w_n = self.para['w_n']
        i_ld = self.time_varying_state['i_ld'][-1]
        i_lq = self.time_varying_state['i_lq'][-1]
        i_ld_star = self.time_varying_state['i_ld_star'][-1]
        i_lq_star = self.time_varying_state['i_lq_star'][-1]
        gamma_d = self.time_varying_state['gamma_d'][-1]
        gamma_q = self.time_varying_state['gamma_q'][-1]
        v_id_star = -w_n*l_f*i_lq + k_pc*(i_ld_star-i_ld)+ k_ic*gamma_d
        v_iq_star =  w_n*l_f*i_ld + k_pc*(i_lq_star-i_lq) + k_ic*gamma_q
        return v_id_star, v_iq_star
