function [phi_d, phi_q] = VolCtr_diff(v_od_star,v_oq_star, v_od0, v_oq0, dt, phi_d0, phi_q0)
%This function updates the state variables in the voltage controllors
%   dt: time step
%   phi_d0, phi_q0: state variables in the last step
%   Euler approach is applied
    dphi_d = v_od_star - v_od0;
    dphi_q = v_oq_star - v_oq0;
    phi_d = phi_d0 + dt* dphi_d;
    phi_q = phi_q0 + dt* dphi_q;
end

