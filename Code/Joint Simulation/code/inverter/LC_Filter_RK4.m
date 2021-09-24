function [i_ld, i_lq, v_od, v_oq] = LC_Filter_RK4(para_LC, i_ld0, i_lq0, v_od0, v_oq0, v_id0, v_iq0, i_od0, i_oq0, w0,dt)
%The function models the dynamics of a LC filter
%   para_LC includes the parameters of the LC filter, e.g., para_LC.r_f
%   the variables ending with 0 are the current step value
%   i_ld, i_lq, v_od, v_oq, i_od, i_oq are next-step update
%   dt: time step
x0 = [i_ld0; i_lq0; v_od0; v_oq0];
[di_ld, di_lq, dv_od, dv_oq] = Dynamics_LC_Filter(para_LC, i_ld0, i_lq0, v_od0, v_oq0, v_id0, v_iq0, i_od0, i_oq0, w0);
k1 = dt*[di_ld; di_lq; dv_od; dv_oq];
x1 = x0 + 0.5*k1;
[di_ld, di_lq, dv_od, dv_oq] = Dynamics_LC_Filter(para_LC, x1(1), x1(2), x1(3), x1(4), v_id0, v_iq0, i_od0, i_oq0, w0);
k2 = dt*[di_ld; di_lq; dv_od; dv_oq];
x2 = x0 + 0.5*k2;
[di_ld, di_lq, dv_od, dv_oq] = Dynamics_LC_Filter(para_LC, x2(1), x2(2), x2(3), x2(4), v_id0, v_iq0, i_od0, i_oq0, w0);
k3 = dt *[di_ld; di_lq; dv_od; dv_oq];
x3 = x0 + k3;
[di_ld, di_lq, dv_od, dv_oq] = Dynamics_LC_Filter(para_LC, x3(1), x3(2), x3(3), x3(4), v_id0, v_iq0, i_od0, i_oq0, w0);
k4 = dt*[di_ld; di_lq; dv_od; dv_oq];
x = x0 + 1/6*(k1 + 2*k2 + 2*k3 +k4);
i_ld = x(1);
i_lq = x(2);
v_od = x(3);
v_oq = x(4);
end

