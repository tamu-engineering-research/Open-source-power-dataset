function [delta, v_od_star, v_oq_star, w, P, Q] = PowCtr_Freq_RK4(v_od,v_oq, i_od, i_oq, para_pc, ...
                                                    P0, Q0, delta0, v_od_star0, w0, w_n, P_star, Q_star,...
                                                    V_star,dt)
%Power Controlle
%   variables ending with 0 are last step variables
%   w_n: nominal frequency
%   v_oq_star == 0;
    Tf = para_pc.Tf;
    Tv = para_pc.Tv;
    Df = para_pc.Df;
    Dv = para_pc.Dv;
    w_c = para_pc.w_c;

%% Dynamics of power sensor
    p_t = v_od*i_od + v_oq*i_oq;
    q_t = -v_od*i_oq + v_oq*i_od;
    % Update P and Q using RK4
    x0 = [P0; Q0];
    [dP,dQ] = Dynamics_PowSensor(p_t, q_t, P0, Q0, w_c);
    k1 = dt*[dP; dQ];
    x1 = x0 + 0.5*k1;
    [dP,dQ] = Dynamics_PowSensor(p_t, q_t, x1(1), x1(2), w_c);
    k2 = dt*[dP; dQ];
    x2 = x0 + 0.5*k2;
    [dP,dQ] = Dynamics_PowSensor(p_t, q_t, x2(1), x2(2), w_c);
    k3 = dt*[dP; dQ];
    x3 = x0 + k3;
    [dP,dQ] = Dynamics_PowSensor(p_t, q_t, x3(1), x3(2), w_c);
    k4 = dt*[dP; dQ];
    x = x0 + 1/6*(k1 + 2*k2 + 2*k3 +k4);
    P = x(1);
    Q = x(2);
    
%% Frequency Droop Control Dynamics
   x0 = [delta0; w0; v_od_star0];
   [ddelta, dw, dv_od_star] = Dyn_Freq_Droop(w0, v_od_star0, P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star);
   % next step
   k1 = dt*[ddelta; dw; dv_od_star];
   x1 = x0 + 0.5*k1;
   [ddelta, dw, dv_od_star] = Dyn_Freq_Droop(x1(2), x1(3), P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star);
   k2 = dt*[ddelta; dw; dv_od_star];
   x2 = x0 + 0.5*k2;
   [ddelta, dw, dv_od_star] = Dyn_Freq_Droop(x2(2), x2(3), P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star);
   k3 = dt*[ddelta; dw; dv_od_star];
   x3 = x0 + k2;
  [ddelta, dw, dv_od_star] = Dyn_Freq_Droop(x3(2), x3(3), P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star);
   k4 = dt*[ddelta; dw; dv_od_star];
   x = x0 + 1/6*(k1 + 2*k2 + 2*k3 +k4);
   delta = x(1);
   w = x(2);
   v_od_star = x(3);
   v_oq_star = 0;
end

