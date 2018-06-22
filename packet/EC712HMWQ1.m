%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%EC712 HMW3 Empirical
%%%Shuowen Chen
%%%Fall 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% i.i.d. 
%%% This section produces asymptotic distribution test statistic when u_t
%%% is iid. Therefore long run and short run variances equal and lambda=0.
N = 1000; % number of steps
Rep = 5000; % number of replications
%preallocation
Weiner = zeros(N,1);
% vanilla
W_star = zeros(N,1); 
W_starsq = zeros(N,1);
% constant
W_starc = zeros(N,1); 
W_starcsq = zeros(N,1);
% time trend
W_starct = zeros(N,1); 
W_starctsq = zeros(N,1);
% store asymptotic distribution for vanilla case
Asymp = zeros(Rep,1);
% store asymp distribution for constant case
Asympc = zeros(Rep,1);
% this is used to regress on Wiener vector
cons = ones(N,1);
% store asymp distribution for const + trend case
Asympct = zeros(Rep,1);
% ths is used to regress on Wiener vector
ct = horzcat(cons, cumsum(ones(N,1)));
for j = 1 : Rep
    e = randn(N,1);
    % generate Weiner as partial sums
    for i = 1 : N
        Weiner(i)=sum(e(1:i),1);
    end 
    % approximate int^1_0 W^(*)(r)dW(r) and int^1_0 W^(*)^2(r)dr       
    for i = 2 : N
        W_star(i) = Weiner(i-1) * e(i);
        W_starsq(i) = (Weiner(i-1))^(2);
    end
    W_approx = sum(W_star)/N;
    W2_approx = sum(W_starsq)/N^(2);
    % get the asymptotic distribution 
    % 1. vanilla regression 
    Asymp(j) = W_approx/W2_approx;
    % 2. const
    % approx W* using res from reg of W_j on cons
    [~,~,res] = regress(Weiner,cons);
    % res is our approximation of W*
    for i = 2 : N
        W_starc(i) = res(i-1)*e(i);
        W_starcsq(i) = (res(i-1))^(2);
    end 
    Wc_approx = sum(W_starc)/N;
    W2c_approx = sum(W_starcsq)/N^(2);
    Asympc(j) = Wc_approx/W2c_approx;
    % 3. const + time trend
    % approx W* using res from reg of W_j on cons + time trend
    [~,~,res2] = regress(Weiner,ct);
    for i = 2 : N
        W_starct(i) = res2(i-1)*e(i);
        W_starctsq(i) = (res2(i-1))^(2);
    end
    Wct_approx = sum(W_starct)/N;
    W2ct_approx = sum(W_starctsq)/N^(2);
    Asympct(j) = Wct_approx/W2ct_approx;
end
%% Finite sample distribution (i.i.d)
% This section produces finite sample distribution of T(alpha_hat-1),
% assuming iid of u_t
alpha = 1;
Ts = [10, 25, 50, 100]; % time periods
T = 0;
% store alpha_hat (vanilla regression)
alphah1 = zeros(Rep, length(Ts));
% regression with constant
alphah2 = zeros(Rep, length(Ts));
% regression with time trend
alphah3 = zeros(Rep, length(Ts));

for i = 1 : length(Ts) % loop over Ts
    for j = 1 : Rep % loop over replications
        T = Ts(i); % Pick ith entry of the Ts as sample size
        % DGP:  unit root 
        e = randn(T,1);
        y = zeros(T+1,1);
    for t = 1 : T
        y(t+1) = y(t) + e(t);
    end
    % discard first entry, which is zero
    y(1) = [];
    % 1. regression: y_t = alpha * y_t-1 + u_t
    X1 = vertcat(NaN, y(1:(end-1)));
    alphah1(j,i) = regress(y,X1);
    % regression: y_t = c + alpha * y_t-1 + u_t
    X2 = horzcat(ones(T,1), X1);
    b = regress(y,X2);
    alphah2(j,i) = b(2); 
    % regression: y_t = c + beta * t + alpha * y_t-1 + u_t
    X3 = horzcat(ones(T,1), cumsum(ones(T,1)), X1);
    bb = regress(y,X3);
    alphah3(j,i) = bb(3);
    end
end
% limiting dist (vanilla)
dist1 = Ts.* (alphah1-1);
% limiting dist (constant)
dist2 = Ts.* (alphah2-1);
% limiting dist (cons + trend)
dist3 = Ts.* (alphah3-1);

%% Plotting
figure('Name', 'T(alphah-1) (iid error + vanilla)');
[y1, x1] = ecdf(Asymp);    [y2, x2] = ecdf(dist1(:,1));   
[y3, x3] = ecdf(dist1(:,2));    [y4, x4] = ecdf(dist1(:,3));
[y5, x5] = ecdf(dist1(:,4));
plot(x1,y1,'r',x2,y2,'-b',x3,y3,'-.b',x4,y4,'--b',x5,y5,':b');
legend('Limiting Dist',strcat(' T = ',num2str(Ts(1))),strcat(' T = ',num2str(Ts(2))),strcat(' T = ',num2str(Ts(3))),...
           strcat(' T = ',num2str(Ts(4))),'Location','NW')
clear x1 x2 x3 x4 x5 y1 y2 y3 y4 y5;
figure('Name', 'T(alphah-1) (iid error + constant)');
[y1, x1] = ecdf(Asympc);    [y2, x2] = ecdf(dist2(:,1));   
[y3, x3] = ecdf(dist2(:,2));    [y4, x4] = ecdf(dist2(:,3));
[y5, x5] = ecdf(dist2(:,4));
plot(x1,y1,'r',x2,y2,'-b',x3,y3,'-.b',x4,y4,'--b',x5,y5,':b');
legend('Limiting Dist',strcat(' T = ',num2str(Ts(1))),strcat(' T = ',num2str(Ts(2))),strcat(' T = ',num2str(Ts(3))),...
           strcat(' T = ',num2str(Ts(4))),'Location','NW')
clear x1 x2 x3 x4 x5 y1 y2 y3 y4 y5;

figure('Name', 'T(alphah-1) (iid error + c + trend)');
[y1, x1] = ecdf(Asympct);    [y2, x2] = ecdf(dist3(:,1));   
[y3, x3] = ecdf(dist3(:,2));    [y4, x4] = ecdf(dist3(:,3));
[y5, x5] = ecdf(dist3(:,4));
plot(x1,y1,'r',x2,y2,'-b',x3,y3,'-.b',x4,y4,'--b',x5,y5,':b');
legend('Limiting Dist',strcat(' T = ',num2str(Ts(1))),strcat(' T = ',num2str(Ts(2))),strcat(' T = ',num2str(Ts(3))),...
           strcat(' T = ',num2str(Ts(4))),'Location','NW')
clear x1 x2 x3 x4 x5 y1 y2 y3 y4 y5;      
%% MA(1)
%%% This section simulates asymptotic distribution of test statistic when
%%% u_t is MA(1): u_t = theta * u_t-1 + e_t
%%% NOTE: this specification adds nuisance parameter lambda into numerator,
%%% yet previous simulations are identical to the iid case. Copying and
%%% pasting is awesome!!
theta = [-0.8, 0.8];
% first column for theta=0.8, second for theta=-0.8
AsympMA = zeros(Rep,2);
AsympMAc = zeros(Rep,2);
AsympMAct = zeros(Rep,2);
% short run variance
var_u = [1+theta(1)^(2), 1+theta(2)^(2)];
% long run variance
var = [(1+theta(1))^(2), (1+theta(2))^(2)];
% nuisance parameter
lambda = [(var(1)-var_u(1))/(2*var(1)), (var(2)-var_u(2))/(2*var(2))];
for l = 1: 2  % loop over thetas
for j = 1 : Rep 
    e = randn(N,1);
    % generate Weiner as partial sums
    for i = 1 : N
        Weiner(i)=sum(e(1:i),1);
    end 
    % approximate int^1_0 W^(*)(r)dW(r) and int^1_0 W^(*)^2(r)dr       
    for i = 2 : N
        W_star(i) = Weiner(i-1) * e(i);
        W_starsq(i) = (Weiner(i-1))^(2);
    end
    W_approx = sum(W_star)/N;
    W2_approx = sum(W_starsq)/N^(2);
    
    % get the asymptotic distribution 
    % 1. vanilla regression 
    AsympMA(j,l) = (W_approx+lambda(l))/W2_approx;
    % 2. const
    % approx W* using res from reg of W_j on cons
    [~,~,res] = regress(Weiner,cons);
    % res is our approximation of W*
    for i = 2 : N
        W_starc(i) = res(i-1)*e(i);
        W_starcsq(i) = (res(i-1))^(2);
    end 
    Wc_approx = sum(W_starc)/N;
    W2c_approx = sum(W_starcsq)/N^(2);
    AsympMAc(j,l) = (Wc_approx + lambda(l))/W2c_approx;
    % 3. const + time trend
    % approx W* using res from reg of W_j on cons + time trend
    [~,~,res2] = regress(Weiner,ct);
    for i = 2 : N
        W_starct(i) = res2(i-1)*e(i);
        W_starctsq(i) = (res2(i-1))^(2);
    end
    Wct_approx = sum(W_starct)/N;
    W2ct_approx = sum(W_starctsq)/N^(2);
    AsympMAct(j,l) = (Wct_approx+lambda(l))/W2ct_approx;
end
end

%% Finite sample distribution (MA(1))
% This section produces finite sample distribution of T(alpha_hat-1),
% assuming MA(1) of u_t
%%% now the regression is an arma(1,1) process
%%% vanilla: y_t = alpha * y_t-1 + e_t + theta * e_t-1
%%% cons: y_t = c + alpha * y_t-1 + e_t + theta * e_t-1
%%% c & trend : y_t = c + beta * t + alpha * y_t-1 + e_t + theta * e_t-1
T2s = [100, 500, 1000]; 
% store alpha_hat (vanilla regression) 
% theta=0.8
alphahma1 = zeros(Rep, length(T2s));
% theta=-0.8
alphahma11 = zeros(Rep, length(T2s));
% regression with constant
alphahma2 = zeros(Rep, length(T2s));
alphahma22 = zeros(Rep, length(T2s));
% regression with time trend
alphahma3 = zeros(Rep, length(T2s));
alphahma33 = zeros(Rep, length(T2s));
for i = 1 : length(T2s) % loop over Ts
    for j = 1 : Rep % loop over replications
        T = T2s(i); % Pick ith entry of the Ts as sample size
        % DGP:  y_t = y_t-1 + u_t
        % u_t = e_t + theta * e_t-1      
        % because there are two theta values, there are two columns of u
        % and y
        e = randn(T,1);
        u = zeros(T,2);
        % assume that e_0 = 0, so that u_1 = e_1
        u(1,1) = e(1);
        u(1,2) = e(1);
        y = zeros(T+1,2);
        % generate u_t
    for t = 2: T
        u(t,1) = e(t) + 0.8 * e(t-1); 
        u(t,2) = e(t) - 0.8 * e(t-1);
    end
        % generate y_t
    for t = 1 : T
        y(t+1,1) = y(t,1) + u(t,1);
        y(t+1,2) = y(t,2) + u(t,2);
    end
    % discard first entries, which are zero
    y(1,:) = [];
    % Now that we have the dgp, let's do some regression...
    % 1. regression: y_t = alpha * y_t-1 + u_t
    X1 = vertcat(NaN, y(1:(end-1),1));
    X1m = vertcat(NaN, y(1:(end-1),2));
    alphahma1(j,i) = regress(y(:,1),X1);
    alphahma11(j,i) = regress(y(:,2),X1m);
    % regression: y_t = c + alpha * y_t-1 + u_t
    X2 = horzcat(ones(T,1), X1);
    X2m = horzcat(ones(T,1), X1m);
    b = regress(y(:,1),X2);
    bm = regress(y(:,2),X2m);
    alphahma2(j,i) = b(2);
    alphahma22(j,i) = bm(2);
    % regression: y_t = c + beta * t + alpha * y_t-1 + u_t
    X3 = horzcat(ones(T,1), cumsum(ones(T,1)), X1);
    X3m = horzcat(ones(T,1), cumsum(ones(T,1)), X1m);
    bb = regress(y(:,1),X3);
    bbm = regress(y(:,2),X3m);
    alphahma3(j,i) = bb(3);
    alphahma33(j,i) = bbm(3);
    end
end
% limiting dist (vanilla)
distma1 = T2s.* (alphahma1-1);
distma11 = T2s.*(alphahma11-1);
% limiting dist (constant)
distma2 = T2s.* (alphahma2-1);
distma22 = T2s.* (alphahma22-1);
% limiting dist (cons + trend)
distma3 = T2s.* (alphahma3-1);
distma33 = T2s.* (alphahma33-1);
%% Plotting
figure('Name', 'T(alphah-1) (ma 0.8 + vanilla)');
[y1, x1] = ecdf(AsympMA(:,2));    [y2, x2] = ecdf(distma1(:,1));   
[y3, x3] = ecdf(distma1(:,2));    [y4, x4] = ecdf(distma1(:,3));
plot(x1,y1,'r',x2,y2,'-b',x3,y3,'-.b',x4,y4,'--b');
legend('Limiting Dist',strcat(' T = ',num2str(T2s(1))),strcat(' T = ',num2str(T2s(2))),strcat(' T = ',num2str(T2s(3))),...
           'Location','NW')
       axis([-15 10 0 1])
clear x1 x2 x3 x4 y1 y2 y3 y4;
figure('Name', 'T(alphah-1) (ma 0.8 + constant)');
[y1, x1] = ecdf(AsympMAc(:,2));    [y2, x2] = ecdf(distma2(:,1));   
[y3, x3] = ecdf(distma2(:,2));    [y4, x4] = ecdf(distma2(:,3));
plot(x1,y1,'r',x2,y2,'-b',x3,y3,'-.b',x4,y4,'--b');
legend('Limiting Dist',strcat(' T = ',num2str(T2s(1))),strcat(' T = ',num2str(T2s(2))),strcat(' T = ',num2str(T2s(3))),...
           'Location','NW')
       axis([-20 10 0 1])
clear x1 x2 x3 x4 y1 y2 y3 y4;

figure('Name', 'T(alphah-1) (ma 0.8 + c + trend)');
[y1, x1] = ecdf(AsympMAct(:,2));    [y2, x2] = ecdf(distma3(:,1));   
[y3, x3] = ecdf(distma3(:,2));    [y4, x4] = ecdf(distma3(:,3));
plot(x1,y1,'r',x2,y2,'-b',x3,y3,'-.b',x4,y4,'--b');
legend('Limiting Dist',strcat(' T = ',num2str(T2s(1))),strcat(' T = ',num2str(T2s(2))),strcat(' T = ',num2str(T2s(3))),...
           'Location','NW')
       axis([-25 10 0 1])
clear x1 x2 x3 x4 y1 y2 y3 y4;    

figure('Name', 'T(alphah-1) (ma -0.8 + vanilla)');
[y1, x1] = ecdf(AsympMA(:,1));    [y2, x2] = ecdf(distma11(:,1));   
[y3, x3] = ecdf(distma11(:,2));    [y4, x4] = ecdf(distma11(:,3));
plot(x1,y1,'r',x2,y2,'-b',x3,y3,'-.b',x4,y4,'--b');
legend('Limiting Dist',strcat(' T = ',num2str(T2s(1))),strcat(' T = ',num2str(T2s(2))),strcat(' T = ',num2str(T2s(3))),...
           'Location','NW')
       axis([-700 0 0 1])
clear x1 x2 x3 x4 y1 y2 y3 y4;  

figure('Name', 'T(alphah-1) (ma -0.8 + c)');
[y1, x1] = ecdf(AsympMAc(:,1));    [y2, x2] = ecdf(distma22(:,1));   
[y3, x3] = ecdf(distma22(:,2));    [y4, x4] = ecdf(distma22(:,3));
plot(x1,y1,'r',x2,y2,'-b',x3,y3,'-.b',x4,y4,'--b');
legend('Limiting Dist',strcat(' T = ',num2str(T2s(1))),strcat(' T = ',num2str(T2s(2))),strcat(' T = ',num2str(T2s(3))),...
           'Location','NW')
       axis([-700 0 0 1])
clear x1 x2 x3 x4 y1 y2 y3 y4;   

figure('Name', 'T(alphah-1) (ma -0.8 + c + trend)');
[y1, x1] = ecdf(AsympMAct(:,1));    [y2, x2] = ecdf(distma33(:,1));   
[y3, x3] = ecdf(distma33(:,2));    [y4, x4] = ecdf(distma33(:,3));
plot(x1,y1,'r',x2,y2,'-b',x3,y3,'-.b',x4,y4,'--b');
legend('Limiting Dist',strcat(' T = ',num2str(T2s(1))),strcat(' T = ',num2str(T2s(2))),strcat(' T = ',num2str(T2s(3))),...
           'Location','NW')
       axis([-1000 0 0 1])
clear x1 x2 x3 x4 y1 y2 y3 y4;    
