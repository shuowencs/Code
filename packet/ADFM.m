function [test,k] = ADFM(y, y_gls)
% This function realizes ADF test. 
% Reference: Chapter 13 page 5. 
% Z_alf = T(alf_hat-1)-X_T;
% Inputs:
% y: inflation data (one country)
%
% Outputs: 
% test: statistic
% k: lag chosen by MAIC
%%%NOTE: this function is very similar to method3, but the results are
%%%different. method3 produces estimate of long run variance, while this
%%%function produces a test. 
T = length(y);
% Specify kmax;
kmax = floor(12*((T/100)^(1/4))); 
MAIC = zeros(kmax, 1); 
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
% The regression we work with is 
% y_t = mu + b_0 * y_t-1 + sum^{k}_{i=1}b_i*Delta y_t-i + e_t 


% OLS: truncation lag is determined by BIC
sigma2 = zeros(kmax,1);
tau_T = zeros(kmax,1);
% Procedure: run OLS for each lag (from 1 to kmax), store each of their
% BIC, pick the one that yields the smallest BIC, then report its
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
% sigma2_h = T^-1*sum^T_{t=k+1}err^2
%sigma2_h = err((k+1):end,1)'*err((k+1):end,1)/T;
[Xk,~] = nestregressor(y,k);
b = regress(y,Xk);
% test is
% T(b_0-1)/(1-sum (b_i))
b_sum = sum(b)-b(1)-b(2);
test = (T*(b(2)-1))/(1-b_sum);
end
