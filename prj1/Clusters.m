if (exist('reportMode', 'var') == 1)
    forReport = true;
else
    clear all;
    close all;
    forReport = false;
    %Possible values: 'leastSqGD', 'leastSq', 'removal', 'removalcor', 
    % 'dummy','ridgeReg';
    stage = '3models';
end;
removingOutliers = false;
enableFullDummyCoding = false;
doPrediction=true;

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

%% 3 models
if (strcmp(stage, '3models'))
  id1 = 57;
  id2 = 42;
  % figure;
  % plot(XTr(idxTr1,id2),yTr(idxTr1),'o',XTr(idxTr2,id2),yTr(idxTr2),'o',XTr(idxTr3,id2),yTr(idxTr3),'o');
  % figure;
  % plot(XTr(idxTr1,id1),yTr(idxTr1),'o',XTr(idxTr2,id1),yTr(idxTr2),'o',XTr(idxTr3,id1),yTr(idxTr3),'o');
  % figure;
  % plot(XTe(idxTe1,id2),yTe(idxTe1),'o',XTe(idxTe2,id2),yTe(idxTe2),'o',XTe(idxTe3,id2),yTe(idxTe3),'o');
  % figure;
  % plot(XTe(idxTe1,id1),yTe(idxTe1),'o',XTe(idxTe2,id1),yTe(idxTe2),'o',XTe(idxTe3,id1),yTe(idxTe3),'o');
  % figure;
  % scatter3(XTr(:,id1),XTr(:,id2),yTr);
  [idxTr,C] = kmeans(horzcat(XTr(:,[id1 id2]),yTr),3);
  C = C(:,[1 2]);
  % figure;
  % plot(XTr(idxTr==1,id2),yTr(idxTr==1),'o',XTr(idxTr==2,id2),yTr(idxTr==2),'o',XTr(idxTr==3,id2),yTr(idxTr==3),'o');
  % figure;
  % plot(XTr(idxTr==1,id1),yTr(idxTr==1),'o',XTr(idxTr==2,id1),yTr(idxTr==2),'o',XTr(idxTr==3,id1),yTr(idxTr==3),'o');
  % figure;
  % plot(XTr(idx1==1,id1),XTr(idx1==1,id2),'o',XTr(idx1==2,id1),XTr(idx1==2,id2),'o',XTr(idx1==3,id1),XTr(idx1==3,id2),'o');
  % 
  % figure;
  % plot(XTr(idx1==1,id2),yTr(idx1==1),'o',XTr(idx1==2,id2),yTr(idx1==2),'o');
  % figure;
  % plot(XTr(idx2==1,id2),yTr(idx2==1),'o',XTr(idx2==2,id2),yTr(idx2==2),'o');
  %% 
  seeds = [1:10 42 43 7500 100500];
  errorTe = zeros(length(seeds), 1);
  errorTr = zeros(length(seeds), 1);
  errorTe3 = zeros(length(seeds), 1);
  errorTr3 = zeros(length(seeds), 1);

  for s = 1:length(seeds)
    seed = seeds(s);

    b = leastSquares(yTr, tXTr);
    b1 = leastSquares(yTr(idxTr==1), tXTr(idxTr==1,:));
    b2 = leastSquares(yTr(idxTr==2), tXTr(idxTr==2,:));
    b3 = leastSquares(yTr(idxTr==3), tXTr(idxTr==3,:));

    errorTe(s) = computeCost( yTe, tXTe, b );
    errorTr(s) = computeCost( yTr, tXTr, b );

    idxTe = whichCluster(C, tXTe);
    errorTe3(s) = computeCost3Clusters( yTe, tXTe, idxTe, b1, b2, b3);
    errorTr3(s) = computeCost3Clusters( yTr, tXTr, idxTr, b1, b2, b3);

  %   figure;
  %   plot(tXTe(:,id2+1),tXTe(:,id1+1),'o');
  %   figure;
  %   plot(XTe(idxTe==1,id2),yTe(idxTe==1),'o',XTe(idxTe==2,id2),yTe(idxTe==2),'o',XTe(idxTe==3,id2),yTe(idxTe==3),'o');
  %   figure;
  %   plot(XTe(idxTe==1,id1),yTe(idxTe==1),'o',XTe(idxTe==2,id1),yTe(idxTe==2),'o',XTe(idxTe==3,id1),yTe(idxTe==3),'o');
  end;

  lSqTestRMSE = sqrt(2*mean(errorTe));
  lSqTrainRMSE = sqrt(2*mean(errorTr));
  lSqTestRMSE3 = sqrt(2*mean(errorTe3));
  lSqTrainRMSE3 = sqrt(2*mean(errorTr3));
  fprintf('Test RMSE:  %d\nTrain RMSE: %d\nTest RMSE3:  %d\nTrain RMSE3: %d\n',lSqTestRMSE,lSqTrainRMSE,lSqTestRMSE3,lSqTrainRMSE3);
end;

%% Ridge regression using normal equations
if (strcmp(stage, 'ridgeReg'))
  disp('Ridge regression using normal equations');
  mvals = [2];
  lvals = logspace(1,5,20);
  
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
      disp('tic');
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
    hy = ylabel('RMSE');
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

%% Figures for report
if(forReport)
  %% Histogram of y
  figure;
  histogram(y_train);
  title('Histogram of y\_train.');
  hx = xlabel('y\_train');
  hy = ylabel('');

  set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
  set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
  grid on;

  set(gcf, 'PaperUnits', 'centimeters');
  set(gcf, 'PaperPosition', [0 0 20 12]);
  set(gcf, 'PaperSize', [20 12]);
  print -dpdf 'report/figures/histY.pdf'

  %% Correlation
  c = corr(X_train,y_train);
  figure;
  plot(c,'o');
  title('Correlation between input variables and output');
  hx = xlabel('Input variable');
  hy = ylabel('Correlation');

  set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
  set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
  set(gcf, 'PaperUnits', 'centimeters');
  set(gcf, 'PaperPosition', [0 0 20 12]);
  set(gcf, 'PaperSize', [20 12]);
  print -dpdf 'report/figures/CorrelationXY.pdf'
  %% 3 clouds
  id1 = 57;
  id2 = 42;
  XTr1 = XTr(yTr<=5500,:);
  yTr1 = yTr(yTr<=5500,:);
  XTr2 = XTr(yTr>5500 & yTr<=10000 & XTr(:,id1)>0.25 & XTr(:,id2)<6.4,:);
  yTr2 = yTr(yTr>5500 & yTr<=10000 & XTr(:,id1)>0.25 & XTr(:,id2)<6.4,:);
  XTr3 = XTr(yTr>=10000,:);
  yTr3 = yTr(yTr>=10000,:);
  figure;
  plot(XTr1(:,id2),yTr1,'o',XTr2(:,id2),yTr2,'o',XTr3(:,id2),yTr3,'o');
  title('Scatter plot of X\_train vs y\_train');
  hx = xlabel('X\_train (column 57)');
  hy = ylabel('y\_train');
  set(gca,'fontsize',13,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
  set([hx; hy],'fontsize',13,'fontname','avantgarde','color',[.3 .3 .3]);
  set(gcf, 'PaperUnits', 'centimeters');
  set(gcf, 'PaperPosition', [0 0 20 12]);
  set(gcf, 'PaperSize', [20 12]);
  print -dpdf 'report/figures/X58vsY.pdf'
  figure;
  plot(XTr1(:,id1),yTr1,'o',XTr2(:,id1),yTr2,'o',XTr3(:,id1),yTr3,'o');
  title('Scatter plot of X\_train vs y\_train');
  hx = xlabel('X\_train (column 42)');
  hy = ylabel('y\_train');
  set(gca,'fontsize',13,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
  set([hx; hy],'fontsize',13,'fontname','avantgarde','color',[.3 .3 .3]);
  set(gcf, 'PaperUnits', 'centimeters');
  set(gcf, 'PaperPosition', [0 0 20 12]);
  set(gcf, 'PaperSize', [20 12]);
  print -dpdf 'report/figures/X43vsY.pdf'
end;

%% Predictions:
if (doPrediction && strcmp(stage, '3models'))
    [XTrn, XTrn_mean, XTrn_std] = normalize(X_train);
    Xtst = adjust(X_test, XTrn_mean, XTrn_std);
    tXTrn = [ones(size(XTrn, 1), 1) XTrn];
    tXtst = [ones(size(Xtst, 1), 1) Xtst];
    [idxTr,C] = kmeans(horzcat(XTrn(:,[42 57]),y_train),3);
    C = C(:,[1 2]);

    b1 = leastSquares(y_train(idxTr==1), tXTrn(idxTr==1,:));
    b2 = leastSquares(y_train(idxTr==2), tXTrn(idxTr==2,:));
    b3 = leastSquares(y_train(idxTr==3), tXTrn(idxTr==3,:));

    idxTe = whichCluster(C, tXtst);
  
    predictions = zeros(size(tXtst,1),1);
    predictions(idxTe==1)= tXtst(idxTe==1,:)*b1;
    predictions(idxTe==2)= tXtst(idxTe==2,:)*b2;
    predictions(idxTe==3)= tXtst(idxTe==3,:)*b3;
    
    csvwrite('predictions_regression.csv', predictions);
    errFile = fopen('test_errors_regression.csv', 'wt');
    fprintf(errFile, 'rmse,%d', lSqTestRMSE3);
    fclose(errFile);
end;