function [ddelta, dv_od_star] = Dyn_Angle_Droop(delta0, v_od_star0, P_star, Q_star, P0, Q0, Ta, Da, Tv, Dv, V_star, delta_star)
%This function defines the dynamics of the frequency droop controller
%   w0, and v_od_star0 are the last step state variables
ddelta = 1/Ta*(Da*(P_star - P0) - delta0 + delta_star);
dv_od_star = 1/Tv*(Dv*(Q_star - Q0) - v_od_star0 + V_star);
end

