function [s2,k]=method2(y)
% This function realizes method 2 in the problem
% Second method: AR BIC
% One of the three methods to estimate long run variance (refer to Chapter 4 Page 8)
% NOTE: this method works for one country. To get results for all 7
% countries, call this function in a for loop. 
%
% Inputs:  
% y: inflation data (1 country)
%
% outputs:
% s2: s^2_(AR,Delta y)(BIC)
% alf_hat: regression estimator, stored for unit root testing
% k: lag used
T=length(y);
% Specify kmax;
kmax=floor(12*((T/100)^(1/4))); 
% The regression we work with is 
% Delta y_t = mu + b_0 * y_t-1 + sum^{k}_{i=1}b_i*Delta y_t-i + e_t
% Let's generate these variables one by one:

% gen y_{t-1}
y_lag = vertcat(NaN,y(1:(end-1))); 
% dependent variable
Delta_y = y - y_lag; 
% generate Delta y_t-i as regressors

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now I realize a problem: regressors change dimensions. So I decided to
% write a function called nestregressor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% OLS: truncation lag is determined by BIC
BIC = zeros(kmax,1);
sigma2 = zeros(kmax,1);
% Procedure: run OLS for each lag (from 1 to kmax), store each of their
% BIC, pick the one that yields the smallest BIC, then report its
% corresponding residuals and regressor estiamtes
for i = 1 : kmax
    [X,~] = nestregressor(y,i);
    [~,~,err] = regress(Delta_y,X);
    sigma2(i)=T^(-1)*(err(i+2:end))'*err(i+2:end);
    BIC(i)=log(sigma2(i))+(log(T)*i)/T;
end

% report lag of k using BIC
[~,k] = min(BIC);
% Now we generate s2_{AR,Delta y} = sigma2_h/(1-sum^k_i=1 b_h)^2
% sigma2_h = T^-1*sum^T_{t=k+1}err^2
%sigma2_h = err((k+1):end,1)'*err((k+1):end,1)/T;
[Xk,~] = nestregressor(y,k);
b = regress(Delta_y,Xk);
% sum up b_hat from 1 to k
b_sum = sum(b)-b(1)-b(2);
s2 = sigma2(k)/(1-b_sum)^(2);
end