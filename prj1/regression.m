if (exist('reportMode', 'var') == 1)
    forReport = true;
else
    clear all;
    close all;
    forReport = false;
    removingOutliers = true;
    enableFullDummyCoding = false;
    %Possible values: 'leastSqGD', 'leastSq', 'removal', 'removalcor', 
    % 'dummy','ridgeReg';
    stage = 'dummy';
end;

load('data/regression.mat');
%% Remove outliers
if(removingOutliers)
  outliers = getOutliers(X_train);
  X_train = X_train(outliers==0,:);
  y_train = y_train(outliers==0);
end;

%% Full dummy coding
if(enableFullDummyCoding)
  X_train = dummyCoding(X_train, [2,12,14,29,48,62]);
end;

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
  rmseTe = zeros(size(mvals,2), size(lvals,2));
  rmseTr = zeros(size(mvals,2), size(lvals,2));
  rmseTT = zeros(size(mvals,2), size(lvals,2));
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
 
    plot(lvals, rmseTe, 'b');
    hold on;
    plot(lvals, rmseTr, 'r');
    set(gca,'XScale', 'log');
    title('Error for the second degree polynom.');
    hx = xlabel('Penalizer coefficient lambda');
    hy = ylabel('rmse');
    legend('Test error', 'Training error', 'Location', 'SouthEast');
    set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
    set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
    grid on;
    
    if (forReport)
      disp('printing the figure');
      set(gcf, 'PaperUnits', 'centimeters');
      set(gcf, 'PaperPosition', [0 0 20 12]);
      set(gcf, 'PaperSize', [20 12]);
      print -dpdf 'report/figures/ridgeRegLoss.pdf'
    end;
  end;
end;

%% Dummy coding
if (strcmp(stage, 'dummy'))
    discFeatures = [2,12,14,29,48,62];
    for i = 1:length(discFeatures)
        dXTr = tXTr;
        dXTe = tXTe;
        for v = unique(XTr(:, discFeatures(i)))'
            dXTr(:,size(dXTr, 2)+1) = (XTr(:,discFeatures(i)) == v);
            dXTe(:,size(dXTe, 2)+1) = (XTe(:,discFeatures(i)) == v);
        end;
        for k = 1:K
            [yTrTe, yTrTr, dXTrTe, dXTrTr] = split4crossValidation(k, idxCV, yTr, dXTr);

            beta = leastSquares(yTrTr, dXTrTr);

            rmseTrSub(k) = computeCost(yTrTr, dXTrTr, beta);
            rmseTeSub(k) = computeCost(yTrTe, dXTrTe, beta);
        end;
        rmseTe(i) = sqrt(2*mean(rmseTeSub));
        rmseTr(i) = sqrt(2*mean(rmseTrSub));
        beta = leastSquares(yTr, dXTr);
        rmseTT(i) = sqrt(2*computeCost(yTe, dXTe, beta));
    end;
    [rmseStar lrmseStar] = min(rmseTe(1,:));
    [rmseTrStar lrmseTrStar] = min(rmseTr(1,:));
    [rmseTTStar lrmseTTStar] = min(rmseTT(1,:));
    fprintf('Test  RMSE: %d\nTrain RMSE: %d\nTT    RMSE: %d\n',rmseStar,rmseTrStar,rmseTTStar);
end;

%% Feature removal
if (strcmp(stage, 'removal'))
    for i = 1:size(tXTr,2)
        errorTeSub = zeros(K, 1);
        errorTrSub = zeros(K, 1);
        rXTr = tXTr(:,[1:i-1 i+1:end]);
        for k = 1:K
            [yTrTe, yTrTr, rXTrTe, rXTrTr] = split4crossValidation(k, idxCV, yTr, rXTr);

            beta = leastSquares(yTrTr, rXTrTr);
            rmseTrSub(k) = computeCost(yTrTr, rXTrTr, beta);
            rmseTeSub(k) = computeCost(yTrTe, rXTrTe, beta);
        end;
        rmseTe(i) = sqrt(2*mean(rmseTeSub));
        rmseTr(i) = sqrt(2*mean(rmseTrSub));
    end
    [rmseStar irmseStar] = min(rmseTe);
    [rmseTrStar irmseTrStar] = min(rmseTr);


    nfrmseBeta = leastSquares(yTr, tXTr(:, [1:irmseStar-1 irmseStar+1:end]));
    nfTestRMSE = sqrt(2*mean(computeCost(yTe, tXTe(:, [1:irmseStar-1 irmseStar+1:end]), nfrmseBeta)));
    fprintf('Test  RMSE: %d\nTrain RMSE: %d\nTT    RMSE: %d\n',rmseStar,rmseTrStar,nfTestRMSE);
end;

%% Feature removal 2
if (strcmp(stage, 'removalcor'))
  c = corr(XTr,yTr);
  sortedc = sort(abs(c));
  for i = 1:size(sortedc,1)
    errorTeSub = zeros(K, 1);
    errorTrSub = zeros(K, 1);
    idx = find(abs(c)>sortedc(i));
    rXTr = tXTr(:,idx);
    for k = 1:K
        [yTrTe, yTrTr, rXTrTe, rXTrTr] = split4crossValidation(k, idxCV, yTr, rXTr);

        beta = leastSquares(yTrTr, rXTrTr);
        rmseTrSub(k) = computeCost(yTrTr, rXTrTr, beta);
        rmseTeSub(k) = computeCost(yTrTe, rXTrTe, beta);
    end;
    rmseTe(i) = sqrt(2*mean(rmseTeSub));
    rmseTr(i) = sqrt(2*mean(rmseTrSub));
  end
  [rmseStar irmseStar] = min(rmseTe);
  [rmseTrStar irmseTrStar] = min(rmseTr);

  nfrmseBeta = leastSquares(yTr, tXTr(:, [1:irmseStar-1 irmseStar+1:end]));
  nfTestRMSE = sqrt(2*mean(computeCost(yTe, tXTe(:, [1:irmseStar-1 irmseStar+1:end]), nfrmseBeta)));
  fprintf('Test  RMSE: %d\nTrain RMSE: %d\nTT    RMSE: %d\n',rmseStar,rmseTrStar,nfTestRMSE);

end;

% if (strcmp(stage, '3models'))
% %% Separate data into three different models
%   % Seperate data into 3 clouds
%   id1 = 58;
%   id2 = 43;
%   figure;
% %   plot(tXTr(:,id2),yTr,'o');
% %   figure;
% %   plot(tXTr(:,id1),yTr,'o');
%   
%   tXTr1 = tXTr(yTr<=5500,:);
%   yTr1 = yTr(yTr<=5500,:);
%   tXTr2 = tXTr(yTr>5500 & yTr<=10000 & tXTr(:,id1)>0.25,:);
%   yTr2 = yTr(yTr>5500 & yTr<=10000 & tXTr(:,id1)>0.25,:);
%   tXTr3 = tXTr(yTr>=10000,:);
%   yTr3 = yTr(yTr>=10000,:);
% %   figure;
% %   plot(tXTr1(:,id2),yTr1,'o',tXTr2(:,id2),yTr2,'o',tXTr3(:,id2),yTr3,'o');
% %   title('Scatter plot of tX\_train vs y\_train');
% %   hx = xlabel('tX\_train (column id2)');
% %   hy = ylabel('y\_train');
% %   set(gca,'fontsize',13,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
% %   set([hx; hy],'fontsize',13,'fontname','avantgarde','color',[.3 .3 .3]);
% %   figure;
% %   plot(tXTr1(:,id1),yTr1,'o',tXTr2(:,id1),yTr2,'o',tXTr3(:,id1),yTr3,'o');
%   %% Least squares using normal equations
%   disp('Least squares using normal equations 3 models');
%   errorTeSub = zeros(K, 1);
%   errorTrSub = zeros(K, 1);
%   errorTeXSelect = zeros(K, 1);
%   th1 = 0:0.1:1;
%   th2 = 1:0.1:2;
%   for i = 1:length(th1)
%     for j = 1:length(th2)
%       for k = 1:K
%           [yTrTe, yTrTr, tXTrTe, tXTrTr] = split4crossValidation(k, idxCV, yTr, tXTr);
%           % Separate into 3 models
%           idxTr1 = find(yTrTr<=5500);
%           idxTe1 = find(yTrTe<=5500);
%           idxTr2 = find(yTrTr>5500 & yTrTr<=10000 & tXTrTr(:,id1)>0.25 & tXTrTr(:,id2)<6.4);
%           idxTe2 = find(yTrTe>5500 & yTrTe<=10000 & tXTrTe(:,id1)>0.25 & tXTrTe(:,id2)<6.4);
%           idxTr3 = find(yTrTr>=10000);
%           idxTe3 = find(yTrTe>=10000);
%           % Display
%           figure;
%           plot(tXTrTr(idxTr1,id2),yTrTr(idxTr1,:),'o',tXTrTr(idxTr2,id2),yTrTr(idxTr2,:),'o',tXTrTr(idxTr3,id2),yTrTr(idxTr3,:),'o');
%           figure;
%           plot(tXTrTe(idxTe1,id2),yTrTe(idxTe1,:),'o',tXTrTe(idxTe2,id2),yTrTe(idxTe2,:),'o',tXTrTe(idxTe3,id2),yTrTe(idxTe3,:),'o');
%           % Compute one beta per model
%           beta1 = leastSquares(yTrTr(idxTr1,:), tXTrTr(idxTr1,:));
%           beta2 = leastSquares(yTrTr(idxTr2,:), tXTrTr(idxTr2,:));
%           beta3 = leastSquares(yTrTr(idxTr3,:), tXTrTr(idxTr3,:));
%           errorTrSub(k) = computeCost3Clouds(yTrTr, tXTrTr, idxTr1, idxTr2, idxTr3, beta1, beta2, beta3);
%           errorTeSub(k) = computeCost3Clouds(yTrTe, tXTrTe, idxTe1, idxTe2, idxTe3, beta1, beta2, beta3);
%           ypred=doPrediction(tXTrTe, beta1, beta2, beta3, th1(i), th2(j), id1, id2);
%           e = yTrTe - ypred;
%           errorTeXSelect(k) = e'*e/(2*length(yTrTe));
%       end
%       err(i,j) = sqrt(2*mean(errorTeXSelect));
%     end
%   end
%   rmseTr = sqrt(2*mean(errorTrSub))
%   rmseTe = sqrt(2*mean(errorTeSub))
%   rmseT = min(min(err))
% 
% end;

%% Predictions:
if (forReport && strcmp(stage, 'leastSq'))
    [XTrn, XTrn_mean, XTrn_std] = normalize(X_train);
    Xtst = adjust(X_test, XTrn_mean, XTrn_std);
    tXTrn = [ones(size(XTrn, 1), 1) XTrn];
    tXtst = [ones(size(Xtst, 1), 1) Xtst];

    beta = leastSquares(y_train, tXTrn);
    predictions = tXtst*beta;
    csvwrite('predictions_regression.csv', predictions);
    errFile = fopen('test_errors_regression.csv', 'wt');
    fprintf(errFile, 'rmse,%d', lSqTestRMSE);
    fclose(errFile);
end;