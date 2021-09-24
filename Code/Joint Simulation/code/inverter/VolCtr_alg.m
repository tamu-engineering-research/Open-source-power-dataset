function [i_ld_star, i_lq_star] = VolCtr_alg(para_vc, i_od, i_oq, v_od, ...
                                    v_oq, v_od_star, v_oq_star, phi_d, phi_q,w_n)
%Algbraic equations for the voltage controler
%   Given all state variables updates, obtain the output variables
    F = para_vc.F;
    C_f = para_vc.C_f;
    K_pv = para_vc.K_pv;
    K_iv = para_vc.K_iv;
    i_ld_star = F*i_od - w_n*C_f*v_oq + K_pv*(v_od_star - v_od) + K_iv*phi_d;
    i_lq_star = F*i_oq + w_n*C_f*v_od + K_pv*(v_oq_star - v_oq) + K_iv*phi_q;
end

