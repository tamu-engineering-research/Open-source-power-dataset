function [dP,dQ] = Dynamics_PowSensor(p_t, q_t, P0, Q0, w_c)
%Power sensor dynamics
%   pt, qt: instantanous power
%   P0, Q0: last step P and Q
dP = -w_c*P0 + w_c*p_t;
dQ = -w_c*Q0 + w_c*q_t;
end

