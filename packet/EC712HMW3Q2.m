%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%EC712 HMW3 Empirical
%%%Shuowen Chen
%%%Problem 2
%%%Fall 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Loading Data
% Load data from Excel file (CAN, DEU, FRA, GBR, ITA, JPN, USA)
% Read Sheet of GDP deflator database
data = xlsread('Database.xls','GDP Deflator Database');  
% logging data
l_data = log(data); 
T = 149;
% Create inflation series for each country
inf = zeros(149,7);
% looping over each country
for j = 1: 7 
    % looping over obs
    for i = 1: 149
        inf(i,j)=400*(l_data(i+1,j)-l_data(i,j));
    end
end
% need this to convert data into matrix of doubles
y = inf;
%% GLS Detrended Data
% We need to GLS detrend the data to do some of the exercises
y_GLS = zeros(149,7);
for i = 1 : 7
    data_GLS = GLS(y(:,i));
    y_GLS(:,i) = data_GLS;
end
%% Method One Estimate of Long Run variance
% The three methods are to estimate long run variance. 
% preallocate
sc1 = zeros(7,1);
bandwidth = zeros(7,1);
% loop over countries to call method1
for i = 1 : 7
    [s2, b] = method1(y(:,i));
    sc1(i,1) = s2;
    bandwidth(i) = b;
end
%% Method Two Estimate
sc2 = zeros(7,1);
k_bic = zeros(7,1);
for i = 1 : 7
    [s2, k] = method2(y(:,i));
    sc2(i,1) = s2;
    k_bic(i) = k;
end
%% Method Three Estimate
sc3 = zeros(7,1);
k_maic = zeros(7,1);
for i = 1 : 7
    [s3, k] = method3(y(:,i),y_GLS(:,i));
    sc3(i,1) = s3;
    k_maic(i) = k;
end
%% (a) PP test: 
Z = zeros(7,3);
sigma2=[sc1 sc2 sc3];
% loop over countries
for i = 1 : 7
    % loop over methods
    for j = 1 : 3
    Z(i,j)=PP(y(:,i),sigma2(i,j));
    end
end
disp('Results of PP test for 7 coutries:')
disp(Z)
%%% If you refer to result by Daeha Cho, Yi Zhang and Zhiteng Zeng, note that
%%% they got UK and Germany orders wrong. Convince yourself by running their
%%% code and mine and tabulate results in the same table. 
%% (b) PP test with GLS detrended data
% method one
sc1_gls = zeros(7,1);
bandwidth_gls = zeros(7,1);
% loop over countries to call method1
for i = 1 : 7
    [s2, b] = method1_gls(y_GLS(:,i));
    sc1_gls(i,1) = s2;
    bandwidth_gls(i) = b;
end
% method two
sc2_gls = zeros(7,1);
k_bic_gls = zeros(7,1);
for i = 1 : 7
    [s2, k] = method2_gls(y_GLS(:,i));
    sc2_gls(i,1) = s2;
    k_bic_gls(i) = k;
end
% method three
sc3_gls = zeros(7,1);
k_maic_gls = zeros(7,1);
for i = 1 : 7
    [s3, k] = method3_gls(y_GLS(:,i));
    sc3_gls(i,1) = s3;
    k_maic_gls(i) = k;
end
Z_GLS = zeros(7,3);
sigma2_GLS=[sc1_gls sc2_gls sc3_gls];
% loop over countries
for i = 1 : 7
    % loop over methods
    for j = 1 : 3
    Z_GLS(i,j)=PP(y_GLS(:,i),sigma2_GLS(i,j));
    end
end
disp('Results of PP test for 7 coutries GLS detrended:')
disp(Z_GLS)
%% (c) MZ_alf test with GLS detrended data
% Refer to Chapter 13 Page 11
MZ_alf=zeros(7,3);
for i = 1 : 7
    for j = 1 : 3
        MZ_alf(i,j) = MZalf(y_GLS(:,i),sigma2_GLS(i,j));
    end
end
% The bandwidth or lags are consistent with previous: essentially inherit
% the respective sigma2's. 
disp('MZ_alpha result for 7 countries inflation with GLS-detrending data')
disp(MZ_alf)
%% (d) ADF and ADF GLS
k_adf_bic = zeros(7,1);
adf_bic = zeros(7,1);
for i=1:7
    [adf_bic(i,1), k_adf_bic(i,1)]=ADF(y(:,i));
end
disp(adf_bic)

k_adf_maic = zeros(7,1);
adf_maic =zeros(7,1);
for i = 1 : 7
    [adf_maic(i,1), k_adf_maic(i,1)]=ADFM(y(:,i),y_GLS(:,i));
end
 
k_adfgls_bic = zeros(7,1);
adfgls_bic = zeros(7,1);
for i = 1 : 7 
   [adfgls_bic(i,1), k_adfgls_bic(i,1)]=ADF_gls(y_GLS(:,i));
end
 
k_adfgls_maic = zeros(7,1);
adfgls_maic =zeros(7,1);
for i = 1 : 7 
     [adfgls_maic(i,1), k_adfgls_maic(i,1)]=ADFM_gls(y_GLS(:,i));
end

ADF_RESULT=[adf_bic adf_maic adfgls_bic adfgls_maic];

disp('ADF AND ADF_GLS with choose trucation with BIC and MAIC result for 7 countries')
disp(ADF_RESULT)
