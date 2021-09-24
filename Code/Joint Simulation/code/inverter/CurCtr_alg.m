function [v_id_star, v_iq_star] = CurCtr_alg(para, w_n, i_ld, i_lq, ...
                                    i_ld_star, i_lq_star, gamma_d, gamma_q)
%Algebraic equations for current controllor

K_pc = para.K_pc;
K_ic = para.K_ic;
L_f = para.L_f;

v_id_star = -w_n*L_f*i_lq + K_pc*(i_ld_star - i_ld) + K_ic*gamma_d;
v_iq_star = w_n*L_f*i_ld + K_pc*(i_lq_star - i_lq) + K_ic*gamma_q;

end

