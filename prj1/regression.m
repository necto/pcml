if (exist('reportMode', 'var') == 1)
    forReport = true;
else
    clear all;
    close all;
    forReport = false;
    %Possible values: 'leastSqGD', 'leastSq', 'removal', 'dummy',
    %'ridgeReg';
    stage = 'ridgeReg';
end;

load('data/regression.mat');

%% Linear regression using gradient descent
if (strcmp(stage, 'leastSqGD'))
  disp('Linear regression using gradient descent');
  seeds = [1:10 42 43 7500 100500];
  errorTe = zeros(length(seeds), 1);
  errorTr = zeros(length(seeds), 1);
  for s = 1:length(seeds)
    seed = seeds(s);
    [XTr, yTr, XTe, yTe] = split(y_train, X_train, 0.9, seed);
    [XTr, XTr_mean, XTr_std] = normalize(XTr);
    XTe = adjust(XTe, XTr_mean, XTr_std);
    tXTr = [ones(size(XTr, 1), 1) XTr];
    tXTe = [ones(size(XTe, 1), 1) XTe];

    lSqGdBeta = leastSquaresGD(yTr, tXTr, 0.6);

    errorTe(s) = computeCost( yTe, tXTe, lSqGdBeta );
    errorTr(s) = computeCost( yTr, tXTr, lSqGdBeta );
  end;

  lSqGdTestRMSE = sqrt(2*mean(errorTe));
  lSqGdTrainRMSE = sqrt(2*mean(errorTr));
  fprintf('Test RMSE:  %d\nTrain RMSE: %d\n',lSqGdTestRMSE,lSqGdTrainRMSE);
end;

%% Least squares using normal equations
if (strcmp(stage, 'leastSq'))
  disp('Least squares using normal equations');
  seeds = [1:10 42 43 7500 100500];
  errorTe = zeros(length(seeds), 1);
  errorTr = zeros(length(seeds), 1);
  for s = 1:length(seeds)
    seed = seeds(s);
    [XTr, yTr, XTe, yTe] = split(y_train, X_train, 0.9, seed);
    [XTr, XTr_mean, XTr_std] = normalize(XTr);
    XTe = adjust(XTe, XTr_mean, XTr_std);
    tXTr = [ones(size(XTr, 1), 1) XTr];
    tXTe = [ones(size(XTe, 1), 1) XTe];

    lSqBeta = leastSquares(yTr, tXTr);

    errorTe(s) = computeCost( yTe, tXTe, lSqBeta );
    errorTr(s) = computeCost( yTr, tXTr, lSqBeta );
  end;

  lSqTestRMSE = sqrt(2*mean(errorTe));
  lSqTrainRMSE = sqrt(2*mean(errorTr));
  fprintf('Test RMSE:  %d\nTrain RMSE: %d\n',lSqTestRMSE,lSqTrainRMSE);
end;

%% Split data into train and validation sets
[XTr, yTr, XTe, yTe] = split(y_train, X_train, 0.9, 42);
% Normalize data
[XTr, XTr_mean, XTr_std] = normalize(XTr);
XTe = adjust(XTe, XTr_mean, XTr_std);
% Form tX
tXTr = [ones(size(XTr, 1), 1) XTr];
tXTe = [ones(size(XTe, 1), 1) XTe];

% Generate indexes for K-fold cross-validation
K = 7;
N = size(yTr, 1);
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K, Nk);
for k = 1:K
    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
end;

%% Ridge regression using normal equations
if (strcmp(stage, 'ridgeReg'))
  disp('Ridge regression using normal equations');
  mvals = [2];
  lvals = logspace(1,5,10);
  
  errorTe = zeros(K, 1);
  errorTr = zeros(K, 1);
  errorTT = zeros(K, 1);
  rmseTe = zeros(size(mvals), size(lvals));
  rmseTr = zeros(size(mvals), size(lvals));
  rmseTT = zeros(size(mvals), size(lvals));
  for j = 1:length(mvals)
    m = mvals(j);
    pXTr = myPoly(tXTr, m);
    pXTe = myPoly(tXTe, m);
    for l = 1:length(lvals)
      lambda = lvals(l);
      for k = 1:K
        [yTrTe, yTrTr, pXTrTe, pXTrTr] = split4crossValidation(k, idxCV, yTr, pXTr);
        
        beta = ridgeRegression(yTrTr, pXTrTr, lambda);
        
        errorTe(k) = computeCost(yTrTe,pXTrTe,beta);
        errorTr(k) = computeCost(yTrTr,pXTrTr,beta);
        errorTT(k) = computeCost(yTe,pXTe,beta);
      end;
      rmseTe(j,l) = sqrt(2*mean(errorTe)); 
      rmseTr(j,l) = sqrt(2*mean(errorTr));
      rmseTT(j,l) = sqrt(2*mean(errorTT));
    end;
    [rmseStar lrmseStar] = min(rmseTe(1,:));
    [rmseTrStar lrmseTrStar] = min(rmseTr(1,:));
    [rmseTTStar lrmseTTStar] = min(rmseTT(1,:));
    fprintf('Test  RMSE: %d\nTrain RMSE: %d\nTT    RMSE: %d\n',rmseStar,rmseTrStar,rmseTTStar);
    figure;
    plot(lvals,rmseTe);
  end;
end;

%% Remove outliers
% outliers = getOutliers(X_train);
% X_train = X_train(outliers==0,:);
% y_train = y_train(outliers==0);

%% Feature removal
c = corr(X_train,y_train);
idx = find(abs(c)>0.04);
X_train = X_train(:,idx);
figure;
plot(corr(X_train,y_train));

%% Enable dummy coding for X_train columns [2,12,14,29,48,62]
% X_train = dummyCoding(X_train, [2,12,14,29,48,61]);

