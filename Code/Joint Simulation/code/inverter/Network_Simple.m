function [P, Q] = Network_Simple(r, L, w_n, V_infD,V_infQ, delta, v)
%Give delta and v, compute power flow
Z = r + 1i*w_n*L;
Y = 1/Z;
G = real(Y);
B = imag(Y);
Y_mag = abs(Y);
sigma = angle(Y);
V = V_infD + 1i*V_infQ;
V_inf_mag = abs(V);
V_inf_ang = angle(V);

% power flow equations
P =  G*v^2 + v*V_inf_mag*(-Y_mag)*cos(delta - V_inf_ang - sigma); % notice the sign: -Y_mag/_sigma is a component of the admittance matrix
Q = -B*v^2 + v*V_inf_mag*(-Y_mag)*sin(delta - V_inf_ang - sigma);
end

