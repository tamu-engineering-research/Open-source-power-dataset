%% Test 1: Investigating  parameters and initial conditions
Sb = 10e3; % base rating VA;
Vb = 220*sqrt(3); % voltage base
f = 50; % 50 Hz system
w_n = 2*pi*f;
r_line1 = 0.35; % Ohm
L_line1 = 0.58/w_n; % H
% include the coupling impedance to the line parameters
Lc = 0.35e-3; % H
r_Lc = 0.03;
r_line = r_line1 + r_Lc;
L_line = L_line1 + Lc;

%% Parameters for LC filter
para.L_f = 1.35e-3;
para.C_f = 50e-6;
para.r_f = 0.1;

%% parameters for power, voltage, & current controllers
para.w_c = 31.41;
para.K_pv = 0.05;
para.K_iv = 390;
para.K_pc = 10.5;
para.K_ic = 16e3;
para.F = 0.75;
para.Tf = 1.5*Sb; % freq droop controller: converting to the actual value (P and V is in p.u., while w is rad/s)
para.Ta = 1;   % angle droop controller: no need to convert to the actual value
para.Tv = 10;  
para.Df = 1.5*Sb;    % previous value 0.8
para.Da = 0.2/Sb;
para.Dv = 0.2/7*Vb/Sb;

%% Initial conditions dispatched by ISO
V_infd = 379; % V
V_infq = -20;  % V
v_od0 = 381.8;
v_oq0 = 0;

%% Initialization
% Power Controller
delta0 = 0;

[i_od0,i_oq0] = LinePlusInfBus(r_line, L_line, w_n, V_infd, ...
                                                  V_infq, v_od0, v_oq0,delta0);
P0 = v_od0*i_od0 + v_oq0*i_oq0;
Q0 = -v_od0*i_oq0 + v_oq0*i_od0;
P_star = P0;
Q_star = Q0;
V_star = v_od0;
w0 = w_n;



% LC filter
i_ld0 = -w_n*para.C_f*v_oq0 + i_od0;
i_lq0 = w_n*para.C_f*v_od0 + i_oq0;
V_idq0 = [para.r_f, -w_n*para.L_f; w_n*para.L_f, para.r_f]*[i_ld0; i_lq0] + [v_od0; v_oq0];
v_id0 = V_idq0(1);
v_iq0 = V_idq0(2);
v_id0_star = v_id0;
v_iq0_star = v_iq0;

% current controller
gamma_d0 = 1/para.K_ic*(v_id0 + w_n*para.L_f*i_lq0);
gamma_q0 = 1/para.K_ic*(v_iq0 - w_n*para.L_f*i_ld0);

% voltage controller
phi_d0 = 1/para.K_iv*(i_ld0 - para.F*i_od0 + w_n*para.C_f*v_oq0);
phi_q0 = 1/para.K_iv*(i_lq0 - para.F*i_oq0 - w_n*para.C_f*v_od0);
                                            

save('Test1_IntCdt.mat', 'P0','Q0','delta0', 'w0','phi_d0', 'phi_q0', 'gamma_d0', 'gamma_q0',...
    'v_od0', 'v_oq0','i_od0', 'i_oq0', 'i_ld0','i_lq0','P_star','Q_star', 'V_star', 'v_id0','v_iq0');
save('Test1_Para.mat','para','w_n','r_line', 'L_line');



