function [data]=GLS(y)
% This function performs GLS detrending to data
% Works for one country
% To detrend all countries, just call this function in a for loop.
% 
% Inputs:
% y: inflation data (one country)
%
% Outputs:
% data: GLS detrended data

% The regression we work with is: 
% y_t = gamma * z_t + v_t; v_t = alpha * v_t-1 + u_t
% NOTE: z_t is 1; this implies bar_alpha = 1 + bar_c/T, where bar_c = -7.
% NOTE: The key assumption is normality of u_t. 
T = length(y);
c_bar = -7;
alf_bar = 1 + c_bar/T;
% To estimate this regression, apply quasi-differencing by multiplying
% (1-alf_bar*L) on both sides: 
% (1-alf_bar*L)y_t = gamma*(1-alf_bar*L)z_t + (1-alf_bar*L)v_t
% This becomes:
% y_t - alf_bar * y_t-1 = gamma * (z_t-alf_bar*z_t-1)+ u_t. Since z_t =1,
% this becomes: 
% y_t - alf_bar * y_t-1 = gamma * (1-alf_bar)+ u_t for t=2:T
% y_t = gamma + u_t for t=1
% Now let's implement this regression, which becomes OLS. 
% Dependent variable is Y, regressor is X
Y = zeros(T,1);
X = ones(T,1);
Y(1) = y(1);
for i = 2 : T
    Y(i) = y(i) - alf_bar * y(i-1);
    X(i) = 1-alf_bar;
end
b = regress(Y,X);
data = y-b;
end
