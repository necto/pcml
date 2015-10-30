if (exist('reportMode', 'var') == 1)
    forReport = true;
else
    forReport = false;
    clear all;
end;

load('data/classification.mat');


y_train = (y_train + 1)/2; % !!! remember invert that for predictions

% dummy-encoding? does not help
%for i = unique(X_train(:,8))'
%    X_train(:,size(X_train, 2)+1) = (X_train(:,8) == i);
%end

[XTr, yTr, XTe, yTe] = split(y_train, X_train, 0.9);
%[XTr, XTr_mean, XTr_std] = normalize(XTr);
%XTe = adjust(XTe, XTr_mean, XTr_std);
tXTr = [ones(size(XTr, 1), 1) XTr];
tXTe = [ones(size(XTe, 1), 1) XTe];

lrBeta = logisticRegression(yTr, tXTr, 1e-6, 1e-5);

lrY = sigmoid(tXTe * lrBeta) > 0.5;
logisticRegressionTestError = sum(lrY ~= yTe)/size(yTe, 1)
logisticRegressionTrainError = sum((sigmoid(tXTr*lrBeta) > 0.5) ~= yTr)/size(yTr, 1)

K = 7;
N = size(yTr, 1);
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K, Nk);
for k = 1:K
    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
end;

%{
for i = 1:size(tXTr,2)
    errorTeSub = zeros(K, 1);
    errorTrSub = zeros(K, 1);
    for k = 1:K
        [yTrTe, yTrTr, tXTrTe, tXTrTr] = split4crossValidation(k, idxCV, yTr, tXTr);
        
        beta = logisticRegression(yTrTr, tXTrTr(:,[1:i-1 i+1:end]), 1e-6, 1e-6);
        errorTeSub(k) = sum((sigmoid(tXTrTe(:,[1:i-1 i+1:end])*beta) > 0.5) ~= yTrTe)/size(yTrTe,1);
        errorTrSub(k) = sum((sigmoid(tXTrTr(:,[1:i-1 i+1:end])*beta) > 0.5) ~= yTrTr)/size(yTrTr,1);
    end
    errorTe(i) = mean(errorTeSub)
    errorTr(i) = mean(errorTrSub);
end
[errorStar iStar] = min(errorTe);

nfBeta = logisticRegression(yTr, tXTr(:, [1:iStar-1 iStar+1:end]), 1e-6, 1e-5);
noFeatureTestError = sum((sigmoid(tXTe(:, [1:iStar-1 iStar+1:end]) * nfBeta) > 0.5) ~= yTe)/size(yTe,1)

plot(errorTe);
%}

%{
mvals = [1 2];
lvals = logspace(-1, 4, 4);

errorTeSub = zeros(K,1);
errorTrSub = zeros(K,1);
for j = 1:length(mvals)
    m = mvals(j);
    pXTr = [ones(size(XTr, 1), 1) myPoly(XTr, m)];
    for l = 1:length(lvals)
        lambda = lvals(l);
        for k = 1:K
            [yTrTe, yTrTr, pXTrTe, pXTrTr] = split4crossValidation(k, idxCV, yTr, pXTr);
            
            if (m == 1)
                beta = penLogisticRegression(yTrTr, pXTrTr, 1e-6, lambda, 1e-5);
            else
                beta = penLogisticRegression(yTrTr, pXTrTr, 1e-12, lambda, 1e-5);
            end;
            errorTeSub(k) = sum((sigmoid(pXTrTe*beta) > 0.5) ~= yTrTe)/size(yTrTe,1);
            errorTrSub(k) = sum((sigmoid(pXTrTr*beta) > 0.5) ~= yTrTr)/size(yTrTr,1);
        end;
        errorTe(j, l) = mean(errorTeSub);
        errorTr(j, l) = mean(errorTrSub)
    end;
end;
%}




% TODO: choose the right low and high borders to produce a nice-looking
% graph.
lvals = logspace(0, 4, 200);

for l = 1:length(lvals)
    lambda = lvals(l);
    
    %errorTeSub = zeros(K,1);
    %errorTrSub = zeros(K,1);
    for k = 1:K
        [yTrTe, yTrTr, tXTrTe, tXTrTr] = split4crossValidation(k, idxCV, yTr, tXTr);
        
        beta = penLogisticRegression(yTrTr, tXTrTr, 1e-6, lambda, 1e-5);
        
        rmseTrSub(k) = logisticRegLoss(beta, tXTrTr, yTrTr);
        rmseTeSub(k) = logisticRegLoss(beta, tXTrTe, yTrTe);
        
        errorTeSub(k) = (sum((sigmoid(tXTrTe*beta) > 0.5) ~= yTrTe))/size(yTrTe,1);
        errorTrSub(k) = (sum((sigmoid(tXTrTr*beta) > 0.5) ~= yTrTr))/size(yTrTr,1);
        errorTTSub(k) = (sum((sigmoid(tXTe*beta) > 0.5) ~= yTe)/size(yTe,1));
    end;
    errorTe(l) = mean(errorTeSub);
    errorTr(l) = mean(errorTrSub);
    errorTT(l) = mean(errorTTSub);
    rmseTr(l) = mean(rmseTrSub);
    rmseTe(l) = mean(rmseTeSub)
end;
[errStar lStar] = min(errorTe);

cvPlrBeta = penLogisticRegression(yTr, tXTr, 1e-6, lvals(lStar), 1e-5);
crossValidationPenalizedLogisticRegressionError = sum((sigmoid(tXTe*beta) > 0.5) ~= yTe)/size(yTe,1)

plot(lvals, errorTe*100, 'b');
%hold on;
%plot(lvals, rmseTe, 'b--');
hold on;
plot(lvals, errorTr*100, 'r');
%hold on;
%plot(lvals, rmseTr, 'r--');
%hold on;
%plot(lvals, errorTT, 'g');
set(gca,'XScale', 'log');
title('Misprediction errors for penalized logistic regression.');
hx = xlabel('Penalizer coefficient lambda');
hy = ylabel('Misprediction fraction (%)');
legend('Test error', 'Training error', 'Location', 'northwest');
set(gca,'fontsize',13,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',13,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;

if (forReport)
    disp('printing the figure');
    set(gcf, 'PaperUnits', 'centimeters');
    set(gcf, 'PaperPosition', [0 0 20 20]);
    set(gcf, 'PaperSize', [20 20]);
    print -dpdf 'report/figures/penLLmisses.pdf'
end;


