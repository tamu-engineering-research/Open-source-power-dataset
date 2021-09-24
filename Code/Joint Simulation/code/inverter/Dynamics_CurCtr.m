function [dgamma_d, dgamma_q] = Dynamics_CurCtr(i_ld_star,i_lq_star, i_ld, i_lq)
%   gamma_d0, gamma_q0: state variables in the last step
%   Euler approach is applied
dgamma_d = i_ld_star - i_ld;
dgamma_q = i_lq_star - i_lq;
end

