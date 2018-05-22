function [Z_alf] = PP(y, sigma2)
% This function realizes PP test. 
% Reference: Chapter 13 section 2. 
% Z_alf = T(alf_hat-1)-X_T;
% Inputs:
% y: inflation data (one country)
% sigma2: the estimate of long run variance (using 3 methods)
%
% Outputs: 
% Z_alf: statistic

T = length(y);
% PP test regression:
% y_t = beta*z_t + alpha * y_t-1 + u_t
% Note: z_t is constant. 
y_lag = vertcat(NaN,y(1:(end-1))); 
% generate X
X = horzcat(ones(T,1),y_lag);
% regression
[b,~,err] = regress(y,X);
% get sigma^2_u
sigma2_u = (err(2:end)'*err(2:end))/T;
% Gen X_T
X_T = (sigma2-sigma2_u)/(2*T^(-2)*((y-mean(y))'*(y-mean(y))));
% Assemble Z_alf
Z_alf = T*(b(2)-1)-X_T;
end