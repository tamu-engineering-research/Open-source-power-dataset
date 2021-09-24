function [di_ld, di_lq, dv_od, dv_oq] = Dynamics_LC_Filter(para_LC, i_ld0, i_lq0, v_od0, v_oq0, v_id0, v_iq0, i_od0, i_oq0, w0)
%The function gives one-step update of the lc filter
%   para_LC includes the parameters of the LC filter, e.g., para_LC.r_f
%   the variables ending with 0 are the current step value
%   i_ld, i_lq, v_od, v_oq, i_od, i_oq are next-step update
%   dt: time step
    di_ld = -para_LC.r_f/para_LC.L_f*i_ld0 + w0 * i_lq0 + 1/para_LC.L_f*(v_id0 - v_od0);
    di_lq = -para_LC.r_f/para_LC.L_f*i_lq0 - w0 * i_ld0 + 1/para_LC.L_f*(v_iq0 - v_oq0);
    dv_od = w0*v_oq0 + 1/para_LC.C_f*(i_ld0 -i_od0);
    dv_oq = -w0*v_od0 + 1/para_LC.C_f*(i_lq0 -i_oq0);
end

