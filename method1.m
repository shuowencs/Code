function [sq,m] = method1(y)
% This function realizes method 1 in the problem
% First method: Quadratic window with Andrew (1991) AR(1)
% One of the three methods to estimate long run variance (refer to Chapter 4 Page 8)
% NOTE: this method works for one country. To get results for all 7
% countries, call this function in a for loop. 
%
% Inputs:  
% y: inflation data (1 country)
%
% outputs:
% sq: s^2_(sc,1)
% alf_hat: regression estimator, stored for unit root testing
% m: bandwidth
%% Get residuals
% The model is  y_t = mu + alpha * y_t-1 + u_t
% H0: alpha = 1; H1: alpha<1
%%% get u_hat: res of regression of model
%%% in the regression, dependent variable is inflation, independent variale
%%% is lag of inflation plus const
y_lag = vertcat(NaN,y(1:(end-1)));
X = horzcat(ones(149,1),y_lag);
[~,~,r] = regress(y,X);
% drop the first because they are NaN
res = r(2:end);
% I am a lazy guy, so instead of calculating autocov by myself, I use
% the formula that autocov = autocorr*var to get the results. Why?
% Because the latter two can be done by Matlab!
T = length(res);
var_res = var(res); % variance of the residuals
autocorr_res = autocorr(res,T-1); % autocorrelation
autocov_res = var_res * autocorr_res(2:end); % autocov
% NOTE: var_res * autocorr_res(1) = var_res trivially, but as specified in the
% problem set, R_h doesn't contain variance term. Therefore I start with
% 2:end. 
%% generate s^2
% specify an ar(1) model
est = arima(1,0,0);
% beta reports regression estimators 
result = estimate(est, y);
rho_h = cell2mat(result.AR); 
% alpha_hat by Andrew (ar(1))
alpha_h = 4*rho_h^(2)/(1-rho_h)^4;
% bandwidth by Andrew 
m = 1.3221*(alpha_h*T)^(1/5);

% preallocation
delta = zeros((T-1),1);
w = zeros((T-1),1);
% loop over 1 to T-1 because index must be positive, but note that this sq
% is symmetric. Need to take care of j=0 case, but note that when j=0, 
% R_hat = var(u), and the weight is 1.
for j = 1:(T-1)
    delta(j) = 6*pi*j/(5*m);
    % quadratic spectral window
    w(j) = 3*(sin(delta(j))/delta(j)-cos(delta(j)))/delta(j)^(2);
end
% get the result
sq = var_res+2*w'*autocov_res;
end