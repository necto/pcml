clear all;
close all;
load('data/regression.mat');

%% Remove outliers
outliers = getOutliers(X_train);
X_train = X_train(outliers==0,:);
y_train = y_train(outliers==0);

%% Enable dummy coding for X_train columns [2,12,14,29,48,62]
X_train = dummyCoding(X_train, [2,12,14,29,48,62]);

%% Split data into train and validation sets
[XTr, yTr, XTe, yTe] = split(y_train, X_train, 0.8);
% Normalize data
[XTr, XTr_mean, XTr_std] = normalize(XTr);
XTe = adjust(XTe, XTr_mean, XTr_std);
% Form tX
tXTr = [ones(size(XTr, 1), 1) XTr];
tXTe = [ones(size(XTe, 1), 1) XTe];

% Generate indexes for K-fold cross-validation
K = 4;
N = size(yTr, 1);
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K, Nk);
for k = 1:K
    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
end;
%{
%% Linear regression using gradient descent
disp('Linear regression using gradient descent');
errorTeSub = zeros(K, 1);
errorTrSub = zeros(K, 1);
for k = 1:K
    [yTrTe, yTrTr, tXTrTe, tXTrTr] = split4crossValidation(k, idxCV, yTr, tXTr);
    beta = leastSquaresGD(yTrTr, tXTrTr, 0.01);
    errorTeSub(k) = computeCost(yTrTe,tXTrTe,beta); 
    errorTrSub(k) = computeCost(yTrTr,tXTrTr,beta); 
end
rmseTr = sqrt(2*mean(errorTrSub))
rmseTe = sqrt(2*mean(errorTeSub))

%% Least squares using normal equations
disp('Least squares using normal equations');
errorTeSub = zeros(K, 1);
errorTrSub = zeros(K, 1);
for k = 1:K
    [yTrTe, yTrTr, tXTrTe, tXTrTr] = split4crossValidation(k, idxCV, yTr, tXTr);
    beta = leastSquares(yTrTr, tXTrTr);
    errorTeSub(k) = computeCost(yTrTe,tXTrTe,beta); 
    errorTrSub(k) = computeCost(yTrTr,tXTrTr,beta); 
end
rmseTr = sqrt(2*mean(errorTrSub))
rmseTe = sqrt(2*mean(errorTeSub))

%% Separate data into three different models
% Seperate data into 3 clouds
tXTr1 = tXTr(yTr<=5500,:);
yTr1 = yTr(yTr<=5500,:);
tXTr2 = tXTr(yTr>5500 & yTr<=10000 & tXTr(:,58)>0.25 & tXTr(:,58)<6.4,:);
yTr2 = yTr(yTr>5500 & yTr<=10000 & tXTr(:,58)>0.25 & tXTr(:,58)<6.4,:);
tXTr3 = tXTr(yTr>=10000,:);
yTr3 = yTr(yTr>=10000,:);
figure;
plot(tXTr1(:,58),yTr1,'o',tXTr2(:,58),yTr2,'o',tXTr3(:,58),yTr3,'o');
title('Scatter plot of tX\_train vs y\_train');
hx = xlabel('tX\_train (column 58)');
hy = ylabel('y\_train');
set(gca,'fontsize',13,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',13,'fontname','avantgarde','color',[.3 .3 .3]);
print -dpdf cluster.pdf
figure;
plot(tXTr1(:,43),yTr1,'o',tXTr2(:,43),yTr2,'o',tXTr3(:,43),yTr3,'o');

%% Least squares using normal equations
disp('Least squares using normal equations 3 models');
errorTeSub = zeros(K, 1);
errorTrSub = zeros(K, 1);
errorTeXSelect = zeros(K, 1);
th1 = 0:0.1:1;
th2 = 1:0.1:2;
for i = 1:length(th1)
  for j = 1:length(th2)
    for k = 1:K
        [yTrTe, yTrTr, tXTrTe, tXTrTr] = split4crossValidation(k, idxCV, yTr, tXTr);

        % Separate into 3 models
        idxTr1 = find(yTrTr<=5500);
        idxTe1 = find(yTrTe<=5500);
        idxTr2 = find(yTrTr>5500 & yTrTr<=10000 & tXTrTr(:,58)>0.25 & tXTrTr(:,58)<6.4);
        idxTe2 = find(yTrTe>5500 & yTrTe<=10000 & tXTrTe(:,58)>0.25 & tXTrTe(:,58)<6.4);
        idxTr3 = find(yTrTr>=10000);
        idxTe3 = find(yTrTe>=10000);
        % Display
%         figure;
%         plot(tXTrTr(idxTr1,58),yTrTr(idxTr1,:),'o',tXTrTr(idxTr2,58),yTrTr(idxTr2,:),'o',tXTrTr(idxTr3,58),yTrTr(idxTr3,:),'o');
%         figure;
%         plot(tXTrTe(idxTe1,58),yTrTe(idxTe1,:),'o',tXTrTe(idxTe2,58),yTrTe(idxTe2,:),'o',tXTrTe(idxTe3,58),yTrTe(idxTe3,:),'o');

        % Compute one beta per model
        beta1 = leastSquares(yTrTr(idxTr1,:), tXTrTr(idxTr1,:));
        beta2 = leastSquares(yTrTr(idxTr2,:), tXTrTr(idxTr2,:));
        beta3 = leastSquares(yTrTr(idxTr3,:), tXTrTr(idxTr3,:));

        errorTrSub(k) = computeCost3Clouds(yTrTr, tXTrTr, idxTr1, idxTr2, idxTr3, beta1, beta2, beta3);
        errorTeSub(k) = computeCost3Clouds(yTrTe, tXTrTe, idxTe1, idxTe2, idxTe3, beta1, beta2, beta3);

        ypred= doPrediction(tXTrTe, beta1, beta2, beta3, th1(i), th2(j));
        e = yTrTe - ypred;
        errorTeXSelect(k) = e'*e/(2*length(yTrTe));
    end
    err(i,j) = sqrt(2*mean(errorTeXSelect));
  end
end
rmseTr = sqrt(2*mean(errorTrSub))
rmseTe = sqrt(2*mean(errorTeSub))
rmseT = min(min(err))
%}
%% Ridge regression using normal equations
disp('Ridge regression using normal equations');
degree = 2;
lvals = logspace(1,4,10);
pXTr = [ones(size(XTr, 1), 1) myPoly(XTr, degree)];
for l = 1:length(lvals)
    lambda = lvals(l);
    
    errorTeSub = zeros(K,1);
    errorTrSub = zeros(K,1);
    for k = 1:K
        [yTrTe, yTrTr, ptXTrTe, ptXTrTr] = split4crossValidation(k, idxCV, yTr, pXTr);
        
        beta = ridgeRegression(yTrTr, ptXTrTr, lambda);
        %yyy = ptXTrTe*beta;
        errorTeSub(k) = computeCost(yTrTe,ptXTrTe,beta);
        errorTrSub(k) = computeCost(yTrTr,ptXTrTr,beta);
    end;
    errorTe(l) = mean(errorTeSub);
    errorTr(l) = mean(errorTrSub);
end;
figure;
plot(lvals,errorTe)

[errTeStar lTeStar] = min(errorTe)
[errTrStar lTrStar] = min(errorTr)