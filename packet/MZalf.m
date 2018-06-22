function test = MZalf(y, sigma2)
% This function realizes MZ_alf test with GLS detrended data. 
% Reference: Chapter 13 section 9. 
% Inputs:
% y: GLS detrended inflation data (one country)
% sigma2: the estimate of long run variance (using 3 methods)
%
% Outputs: 
% test: statistic
%
%
% MZ^GLS_alf = (T^(-1)y^2_T-sigma2)/(2T^(-2)sum^T_{t=1}y^2_{t-1});
T = length(y);
y_lag = vertcat(NaN,y(1:(end-1)));
sum = (y_lag(2:end))'*(y_lag(2:end));
test = (T^(-1)*y(T)^(2)-sigma2)/(2*T^(-2)*sum);
end
