function [gamma_d, gamma_q] = CurCtr_diff(i_ld_star,i_lq_star, i_ld, i_lq, dt, gamma_d0, gamma_q0)
%This function updates the state variables in the current controllors
%   dt: time step
%   gamma_d0, gamma_q0: state variables in the last step
%   Euler approach is applied
dgamma_d = i_ld_star - i_ld;
dgamma_q = i_lq_star - i_lq;
gamma_d = gamma_d0 + dgamma_d*dt;
gamma_q = gamma_q0 + dgamma_q*dt;
end

