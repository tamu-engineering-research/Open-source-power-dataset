function [delta,w,v] = PowCtr_Freq_Simple(delta0, w0, v0, P0, Q0, ...
                                   V_star, P_star, Q_star, para,dt, w_n)
%This is a simple version of the angle droop controller; inner loop
%controllers are not modeled in this function.
Tf = para.Tf;
Df = para.Df;
Tv = para.Tv;
Dv = para.Dv;


% Update delta and v using RK4
    x0 = [delta0; w0; v0];
   [ddelta, dw, dv] = Dyn_Freq_Droop_Simple( x0(2), x0(3), P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star);
    k1 = dt*[ddelta; dw; dv];
    x1 = x0 + 0.5*k1;
    [ddelta, dw, dv] = Dyn_Freq_Droop_Simple(x1(2), x1(3), P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star);
    k2 = dt*[ddelta; dw; dv];
    x2 = x0 + 0.5*k2;
    [ddelta, dw, dv] = Dyn_Freq_Droop_Simple(x2(2), x2(3), P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star);
    k3 = dt*[ddelta;dw; dv];
    x3 = x0 + k3;
    [ddelta, dw, dv] = Dyn_Freq_Droop_Simple(x3(2), x3(3), P_star, Q_star, P0, Q0, Tf, Df, Tv, Dv, w_n, V_star);
    k4 = dt*[ddelta; dw; dv];
    x = x0 + 1/6*(k1 + 2*k2 + 2*k3 +k4);
    delta = x(1);
    w = x(2);
    v = x(3);
end

