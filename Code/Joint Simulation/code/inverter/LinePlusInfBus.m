function [i_d,i_q] = LinePlusInfBus(r, L, w_n, V_infD, ...
                                                  V_infQ, v_bd, v_bq, delta)
% Update current from the network via algebraic equations
% ZI = V
% with DQ transform
Z = [r, -w_n*L; +w_n*L, +r];
T = [cos(delta), -sin(delta); sin(delta), cos(delta)]; %dq to DQ
T_inv = [cos(delta), sin(delta); -sin(delta), cos(delta)];
v_DQ = T*[v_bd; v_bq];
v_bD = v_DQ(1);
v_bQ = v_DQ(2);
I = inv(Z)*[v_bD - V_infD; v_bQ - V_infQ];
I_dq = T_inv*I;
i_d = I_dq(1);
i_q = I_dq(2);

end

