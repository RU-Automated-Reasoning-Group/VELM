function [dx]=car_dynamics(x,u)
 

% lead car dynamics

dx(1, 1) = x(3);
dx(2, 1) = x(4);
dx(3, 1) = 2.0 * 0.001027 * x(4) + 3 * 0.001027 * 0.001027 * x(1) + u(1) / 12.;
dx(4, 1) = -2.0 * 0.001027 * x(3) + u(3) / 12.;
dx(5, 1) = ((2.0 * 0.001027 * x(4) + 3 * 0.001027 * 0.001027 * x(1) + u(1) / 12.) * x(3) + (-2.0 * 0.001027 * x(3) + u(3) / 12.) * x(4)) / x(5);
dx(6, 1) = 2.0 * 0.001027 * (x(1) * x(3) + x(2) * x(4)) / sqrt(x(1) * x(1) + x(2) * x(2));


           