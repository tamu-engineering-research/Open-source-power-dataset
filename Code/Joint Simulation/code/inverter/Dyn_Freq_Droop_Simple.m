function [ddelta, dw, dv] = Dyn_Freq_Droop_Simple(w0, v, P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star)
%This function defines the dynamics of the frequency droop controller
%   w0, and v_od_star0 are the last step state variables
ddelta = w0 - w_n;
dw = 1/Tf*(-Df*(w0 - w_n) + P_star - P0);
dv = 1/Tv*(Dv*(Q_star - Q0) - v + V_star);
end

