%% Test 5: This experiment introduces a three-phase to ground fault
clear;
load Test1_IntCdt.mat;
load Test1_Para.mat;
dt = 0.00001;
t_end = 10; % simulation time
t = 0:dt:t_end;
m = length(t);

%% Create a 3-phi to ground fault
V_infd = 379*ones(m,1); % V
V_infq = -20*ones(m,1);  % V
t_fault_start = 1; % fault happens at 5 second
t_dur = 2000*dt*3; % 3 cycle
fault_seg = V_infd(t>=t_fault_start & t<=(t_fault_start+t_dur));
V_infd(t>=t_fault_start & t<=(t_fault_start+t_dur)) = zeros(length(fault_seg),1);
V_infq(t>=t_fault_start & t<=(t_fault_start+t_dur)) = zeros(length(fault_seg),1);
%% create array to store data
delta = zeros(m,1);         delta(1) = delta0;
w = zeros(m,1);             w(1) = w0;
P = zeros(m,1);             Q = zeros(m,1);     
P(1) = P0;                  Q(1) = Q0; 
phi_d = zeros(m,1);         phi_q = zeros(m,1);
phi_d(1) = phi_d0;          phi_q(1) = phi_q0;
i_ld_star = zeros(m,1);     i_lq_star = zeros(m,1);
i_ld_star(1) = i_ld0;       i_lq_star(1) = i_lq0;
gamma_d = zeros(m,1);       gamma_q = zeros(m,1);
gamma_d(1) = gamma_d0;      gamma_q(1) = gamma_q0;
v_id = zeros(m,1);     v_iq = zeros(m,1);
v_id(1) = v_id0;       v_iq(1) = v_iq0;
i_ld = zeros(m,1);          i_lq = zeros(m,1);
i_ld(1) = i_ld0;            i_lq(1) = i_lq0;
v_od = zeros(m,1);          v_oq = zeros(m,1);
v_od(1) = v_od0;            v_oq(1) = v_oq0;
v_od_star = v_od;           v_oq_star = v_od;
i_od = zeros(m,1);          i_oq = zeros(m,1);
i_od(1) = i_od0;            i_oq(1) = i_oq0; 
%% do simulation
seg_num = 20; % every 5% iteration, output an indicator
seg_k = floor(m/seg_num);
for k = 2:m
    %% Next-step prediction using differential equation
    % Power Controler
    [delta(k), v_od_star(k), v_oq_star(k), w(k), P(k), Q(k)] = PowCtr_Freq_RK4(v_od(k-1),v_oq(k-1), i_od(k-1), i_oq(k-1), para, ...
                                                    P(k-1), Q(k-1), delta(k-1), v_od_star(k-1), w(k-1), w_n, P_star, Q_star,...
                                                    V_star,dt);
    % Voltage Controler
    [phi_d(k), phi_q(k)] = VolCtr_diff(v_od_star(k-1),v_oq_star(k), v_od(k-1), v_oq(k-1), dt, phi_d(k-1), phi_q(k-1));
    % Current Controler
    [gamma_d(k), gamma_q(k)] = CurCtr_diff(i_ld_star(k-1),i_lq_star(k-1), i_ld(k-1), i_lq(k-1), dt, gamma_d(k-1), gamma_q(k-1));
    % LC filter
    [i_ld(k), i_lq(k), v_od(k), v_oq(k)] = LC_Filter_RK4(para, i_ld(k-1), i_lq(k-1), v_od(k-1), v_oq(k-1), v_id(k-1), v_iq(k-1), i_od(k-1), i_oq(k-1),w(k-1),dt);
    %% Update algebraic variables using network equations
    [i_od(k),i_oq(k)] = LinePlusInfBus(r_line, L_line, w_n, V_infd(k), ...
                                                  V_infq(k), v_od(k), v_oq(k), delta(k));
    [i_ld_star(k), i_lq_star(k)] = VolCtr_alg(para, i_od(k), i_oq(k), v_od(k), ...
                                    v_oq(k), v_od_star(k), v_oq_star(k), phi_d(k), phi_q(k),w_n);
    [v_id(k), v_iq(k)] = CurCtr_alg(para, w_n, i_ld(k), i_lq(k), ...
                                    i_ld_star(k), i_lq_star(k), gamma_d(k), gamma_q(k));
    %% Print an indicator suggesting the simulation progress
    if rem(k, seg_k) ==0
        fprintf('Current Progess %d percent\n', k/seg_k*5);
    end
end
figure;
plot(t, i_ld,'LineWidth',1);
xlabel('time (sec)');
ylabel('i_ld');
grid on;
figure;
plot(t, i_lq,'LineWidth',1);
xlabel('time (sec)');
ylabel('i_lq');
grid on;

figure;
plot(t, v_od,'LineWidth',1);
xlabel('time (sec)');
ylabel('v_od');
grid on;
figure;
plot(t, v_oq,'LineWidth',1);
xlabel('time (sec)');
ylabel('v_oq');
grid on;

figure;
plot(t, P,'LineWidth',1);
xlabel('time (sec)');
ylabel('P');
grid on;
figure;
plot(t, Q,'LineWidth',1);
xlabel('time (sec)');
ylabel('Q');
grid on;

figure;
plot(t, delta,'LineWidth',1);
xlabel('time (sec)');
ylabel('delta');
grid on;
figure;
plot(t, w,'LineWidth',1);
xlabel('time (sec)');
ylabel('w');
grid on;

save('Test5_DetailedModel_Freq.mat', 'w', 'delta', 'P','Q','i_ld','i_lq','v_od','v_oq','t');

