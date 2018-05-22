function [test,k] = ADF(y)
% This function realizes ADF test. 
% Reference: Chapter 13 page 5. 
% Inputs:
% y: inflation data (one country)
%
% Outputs: 
% test: statistic
% k: lag chosen by BIC
%%%NOTE: this function is very similar to method2, but the results are
%%%different. method2 produces estimate of long run variance, while this
%%%function produces a test. 
T = length(y);
% Specify kmax;
kmax = floor(12*((T/100)^(1/4))); 
BIC = zeros(kmax, 1); 
% gen y_{t-1}
y_lag = vertcat(NaN,y(1:(end-1))); 
% dependent variable
Delta_y = y - y_lag; 
% generate Delta y_t-i as regressors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now I realize a problem: regressors change dimensions. So I decided to
% write a function called nestregressor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Two regressions: 
% y_t = mu + alf * y_t-1 + sum^{k}_{i=1}b_i*Delta y_t-i + e_t
%%% NOTE: this specification is identical to the one in page 5. An alternative is to
%%% subtract y_t-1 from both sides:
%%% Delta y_t = mu + alf * y_t-1 + sum^{k}_{i=1}b_i*Delta y_t-i + e_t
%%% However, the null transforms from alf = 1 to alf = 0, when null is alf=1,
%%% test is as specified in page 5. If we use transform case, resort to
%%% the usual t test. 


% OLS: truncation lag is determined by BIC
sigma2 = zeros(kmax,1);
% Procedure: run OLS for each lag (from 1 to kmax), store each of their
% BIC, pick the one that yields the smallest BIC, then report its
% corresponding residuals and regressor estiamtes
for i = 1 : kmax
    [X,~] = nestregressor(y,i);
    [~,~,err] = regress(y,X);
    sigma2(i)=T^(-1)*(err(i+2:end))'*err(i+2:end);
    BIC(i)=log(sigma2(i))+(log(T)*i)/T;
end

% report lag of k using BIC
[~,k] = min(BIC);
[Xk,~] = nestregressor(y,k);
b = regress(y,Xk);
% t test
b_sum = sum(b)-b(1)-b(2);
test = (T*(b(2)-1))/(1-b_sum);
end