function [ddelta, dw, dv_od_star] = Dyn_Freq_Droop(w0, v_od_star0, P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star)
%This function defines the dynamics of the frequency droop controller
%   w0, and v_od_star0 are the last step state variables
ddelta = w0 - w_n;
dw = 1/Tf*(-Df*(w0 - w_n) + P_star - P0);
dv_od_star = 1/Tv*(Dv*(Q_star - Q0) - v_od_star0 + V_star);
end

