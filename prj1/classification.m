clear all;
load('data/classification.mat');

y_train = (y_train + 1)/2; % !!! remember invert that for predictions

[XTr, yTr, XTe, yTe] = split(y_train, X_train, 0.7);
[XTr, XTr_mean, XTr_std] = normalize(XTr);
XTe = adjust(XTe, XTr_mean, XTr_std);
tXTr = [ones(size(XTr, 1), 1) XTr];
tXTe = [ones(size(XTe, 1), 1) XTe];

lrBeta = logisticRegression(yTr, tXTr, 0.001);

lrY = sigmoid(tXTe * lrBeta) > 0.5;
logisticRegressionTestError = sum(lrY ~= yTe)/size(yTe, 1)

K = 4;
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
        
        beta = logisticRegression(yTrTr, tXTrTr(:,[1:i-1 i+1:end]), 0.001);
        errorTeSub(k) = sum((sigmoid(tXTrTe(:,[1:i-1 i+1:end])*beta) > 0.5) ~= yTrTe)/size(yTrTe,1);
        errorTrSub(k) = sum((sigmoid(tXTrTr(:,[1:i-1 i+1:end])*beta) > 0.5) ~= yTrTr)/size(yTrTr,1);
    end
    errorTe(i) = mean(errorTeSub)
    errorTr(i) = mean(errorTrSub);
end
[errorStar iStar] = min(errorTe);

nfBeta = logisticRegression(yTr, tXTr(:, [1:iStar-1 iStar+1:end]), 0.001);
noFeatureTestError = sum((sigmoid(tXTe(:, [1:iStar-1 iStar+1:end]) * nfBeta) > 0.5) ~= yTe)/size(yTe,1)

plot(errorTe);
%}

mvals = [1 2];
lvals = logspace(-2, 2, 7);

errorTeSub = zeros(K,1);
errorTrSub = zeros(K,1);
for j = 1:length(mvals)
    m = mvals(j);
    pXTr = [ones(size(XTr, 1), 1) myPoly(XTr, m)];
    for l = 1:length(lvals)
        lambda = lvals(l);
        for k = 1:K
            [yTrTe, yTrTr, pXTrTe, pXTrTr] = split4crossValidation(k, idxCV, yTr, pXTr);
        
            beta = penLogisticRegression(yTrTr, pXTrTr, 0.001, lambda);
            errorTeSub(k) = sum((sigmoid(pXTrTe*beta) > 0.5) ~= yTrTe)/size(yTrTe,1);
            errorTrSub(k) = sum((sigmoid(pXTrTr*beta) > 0.5) ~= yTrTr)/size(yTrTr,1);
        end;
        errorTe(j, l) = mean(errorTeSub);
        errorTr(j, l) = mean(errorTrSub)
    end;
end;

%{

% TODO: choose the right low and high borders to produce a nice-looking
% graph.
lvals = logspace(-1, 1, 100);

for l = 1:length(lvals)
    lambda = lvals(l);
    
    errorTeSub = zeros(K,1);
    errorTrSub = zeros(K,1);
    for k = 1:K
        [yTrTe, yTrTr, tXTrTe, tXTrTr] = split4crossValidation(k, idxCV, yTr, tXTr);
        
        beta = penLogisticRegression(yTrTr, tXTrTr, 0.001, lambda);
        
        errorTeSub(k) = (sum((sigmoid(tXTrTe*beta) > 0.5) ~= yTrTe))/size(yTrTe,1);
        errorTrSub(k) = (sum((sigmoid(tXTrTr*beta) > 0.5) ~= yTrTr))/size(yTrTr,1);
    end;
    errorTe(l) = mean(errorTeSub);
    errorTr(l) = mean(errorTrSub);
end;
[errStar lStar] = min(errorTe);

cvPlrBeta = penLogisticRegression(yTr, tXTr, 0.001, lvals(lStar));
crossValidationPenalizedLogisticRegressionError = sum((sigmoid(tXTe*beta) > 0.5) ~= yTe)/size(yTe,1)

plot(lvals, errorTe);
hold on;
plot(lvals, errorTr);
set(gca,'XScale', 'log');

%}

