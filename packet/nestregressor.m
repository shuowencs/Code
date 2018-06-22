function [X,Xb] = nestregressor(y, j)   
% The function does the following thing: when lag is j, the X is 
% (mu, y_t-1, Delta y_t-1,...,Delta y_t-j)
% Inputs:
% j : number of lags
% Outputs:
% X : regressors
T = length(y);
y_lag = vertcat(NaN,y(1:(end-1))); 
Delta_yt = zeros(T,j);
for l = 1 : j
  % gen y_{t-i}
  y_Li = vertcat(NaN(l,1),y(1:(end-l)));
  % gen y_{t-i-1}
  y_Lii = vertcat(NaN(l+1,1),y(1:(end-l-1)));
  % gen Delta y_{t-i}
  Delta_yt(:,l) = y_Li - y_Lii;
end
% generate X
X = horzcat(ones(T,1),y_lag, Delta_yt);
% generate Xb (used in part(b))
Xb = horzcat(y_lag, Delta_yt);
end
