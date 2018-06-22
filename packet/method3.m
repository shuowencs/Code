function [s2, k] = method3(y,y_gls)
% This function realizes method 3 in the problem
% Third method: AR MAIC (Modified AIC)
% One of the three methods to estimate long run variance (refer to Chapter 4 Page 8)
% NOTE: this method works for one country. To get results for all 7
% countries, call this function in a for loop. 
%
% Inputs:  
% y: inflation data (1 country)
% NOTE: I wrote another function to implement GLS detrending. 
%
% Outputs:
% s2: s^2_(AR,Delta y)(MAIC)
% alf_hat: regression estimator, stored for unit root testing
% k: lag used
%
% The regression we work with is 
% Delta y_t = mu + b_0 * y_t-1 + sum^{k}_{i=1}b_i*Delta y_t-i + e_t
% 
% MAIC (Chapter 13 Page 15):
% MAIC(k)=log(u_hat^2_ek)+2*(tau_T(k)+k)/T; 
% tau_T(k) = inv(u_hat^2_ek)*(b0_hat)^2*T^-2*sum^T_t=1(tilde y_t-1)^2

T=length(y);
% Specify kmax;
kmax=floor(12*((T/100)^(1/4)));
% gen y_{t-1}
y_lag = vertcat(NaN,y(1:(end-1))); 
% gen y_gls lag
y_gls_lag = vertcat(NaN,y_gls(1:(end-1)));
% dependent variable
Delta_y = y - y_lag; 
% generate Delta y_t-i as regressors

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now I realize a problem: regressors change dimensions. So I decided to
% write a function called nestregressor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% OLS: truncation lag is determined by BIC
MAIC = zeros(kmax,1);
sigma2 = zeros(kmax,1);
tau_T = zeros(kmax,1);
% Procedure: run OLS for each lag (from 1 to kmax), store each of their
% MAIC, pick the one that yields the smallest MAIC, then report its
% corresponding residuals and regressor estiamtes
for i = 1 : kmax
    [X,~] = nestregressor(y,i);
    [b,~,err] = regress(Delta_y,X);
    sigma2(i)=T^(-1)*(err(i+2:end))'*err(i+2:end);
    % MAIC(k)=log(u_hat^2_ek)+2*(tau_T(k)+k)/T; 
    % tau_T(k) = inv(u_hat^2_ek)*(b0_hat)^2*T^-2*sum^T_t=1(y_t-1)^2
    tau_T(i) = sigma2(i)^(-1)*b(2)^(2)*T^(-2)*((y_gls_lag(2:end))'*y_gls_lag(2:end));
    % NOTE: y_lag(1) is NaN, so didn't include it. This is equivalent to
    % the assumption that y_0 = 0. 
    MAIC(i)=log(sigma2(i))+2*(tau_T(i)+i)/T;
end

% report lag of k using BIC
[~,k] = min(MAIC);
% Now we generate s2_{AR,Delta y} = sigma2_h/(1-sum^k_i=1 b_h)^2
[Xk,~] = nestregressor(y,k);
b = regress(Delta_y,Xk);
% sum up b_hat from 1 to k
b_sum = sum(b)-b(1)-b(2);
s2 = sigma2(k)/(1-b_sum)^(2);
end
%%%%%%%%%%%%%%
%%% If you refer to code by Daeha Cho, Yi Zhang and Zhiteng Zeng, note that
%%% their end result would be 
%%% s2 = sigma2(k)/(1-b(1:end))^(2);
%%% My result is s2 = sigma2(k)/(1-b(3:end))^(2);
%%% I believe my version is current. If MAIC is doing a worse job than BIC,
%%% why modify it in the fist place?
