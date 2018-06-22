function [test,k] = ADF_gls(y)
% This function realizes ADF test using GLS data. 
% Reference: Chapter 13 page 5. 
% Inputs:
% y: gls detrended inflation data (one country)
%
% Outputs: 
% test: statistic
% k: lag chosen by BIC
T=length(y);
% Specify kmax;
kmax=floor(12*((T/100)^(1/4))); 
% The regression we work with is 
% y_t = b_0 * y_t-1 + sum^{k}_{i=1}b_i*Delta y_t-i + e_t
BIC = zeros(kmax,1);
sigma2 = zeros(kmax,1);
% gen y_{t-1}
y_lag = vertcat(NaN,y(1:(end-1))); 
% dependent variable
Delta_y = y - y_lag; 
% Procedure: run OLS for each lag (from 1 to kmax), store each of their
% BIC, pick the one that yields the smallest BIC, then report its
% corresponding residuals and regressor estiamtes
for i = 1 : kmax
    [~,X] = nestregressor(y,i);
    [~,~,err] = regress(Delta_y,X);
    sigma2(i)=T^(-1)*(err(i+1:end))'*err(i+1:end);
    BIC(i)=log(sigma2(i))+(log(T)*i)/T;
end
% report lag of k using BIC
[~,k] = min(BIC);
[~,Xk] = nestregressor(y,k);
b = regress(y,Xk);
% test is
% T(b_0-1)/(1-sum (b_i))
b_sum = sum(b)-b(1);
test = (T*(b(1)-1))/(1-b_sum);
end
