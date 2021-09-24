function [delta,v, w] = PowCtr_Ang_Simple(delta0, v0, P, Q, ...
                                   delta_star, V_star, P_star, Q_star, para,dt, w_n)
%This is a simple version of the angle droop controller; inner loop
%controllers are not modeled in this function.
Ta = para.Ta;
Da = para.Da;
Tv = para.Tv;
Dv = para.Dv;


% Update delta and v using RK4
    x0 = [delta0; v0];
   [ddelta, dv] = Dyn_Angle_Droop_Simple(x0(1), x0(2), P, Q, ...
                                   delta_star, V_star, P_star, Q_star,...
                                   Ta, Da, Tv, Dv);
    k1 = dt*[ddelta; dv];
    x1 = x0 + 0.5*k1;
    [ddelta, dv] = Dyn_Angle_Droop_Simple(x1(1), x1(2), P, Q, ...
                                   delta_star, V_star, P_star, Q_star,...
                                   Ta, Da, Tv, Dv);
    k2 = dt*[ddelta; dv];
    x2 = x0 + 0.5*k2;
    [ddelta, dv] = Dyn_Angle_Droop_Simple(x2(1), x2(2), P, Q, ...
                                   delta_star, V_star, P_star, Q_star,...
                                   Ta, Da, Tv, Dv);
    k3 = dt*[ddelta; dv];
    x3 = x0 + k3;
    [ddelta, dv] = Dyn_Angle_Droop_Simple(x3(1), x3(2), P, Q, ...
                                   delta_star, V_star, P_star, Q_star,...
                                   Ta, Da, Tv, Dv);
    k4 = dt*[ddelta; dv];
    x = x0 + 1/6*(k1 + 2*k2 + 2*k3 +k4);
    delta = x(1);
    v = x(2);
    [ddelta, ~] = Dyn_Angle_Droop_Simple(delta, v, P, Q, ...
                                   delta_star, V_star, P_star, Q_star,...
                                   Ta, Da, Tv, Dv);
    w = w_n + ddelta;
end

