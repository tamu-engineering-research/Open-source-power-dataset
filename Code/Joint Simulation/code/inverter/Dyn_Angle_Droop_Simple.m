function [ddelta, dv] = Dyn_Angle_Droop_Simple(delta0, v0, P, Q, ...
                                   delta_star, V_star, P_star, Q_star,...
                                   Ta, Da, Tv, Dv)
%Differential equation for the angle droop control
    ddelta = 1/Ta*(Da*(P_star-P)-(delta0-delta_star));
    dv = 1/Tv*(Dv*(Q_star - Q)-(v0 - V_star));
end

