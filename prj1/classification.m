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
for k = 1:K
    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
end;

% TODO: choose the right low and high borders to produce a nice-looking
% graph.
lvals = logspace(-1, 1, 100);

for l = 1:length(lvals)
    lambda = lvals(l);
    
    errorTeSub = zeros(K,1);
    errorTrSub = zeros(K,1);
    for k = 1:K
        idxTe = idxCV(k, :);
        idxTr = idxCV([1:k-1 k+1:end]);
        idxTr = idxTr(:);
        yTrTe = yTr(idxTe);
        yTrTr = yTr(idxTr);
        tXTrTe = tXTr(idxTe,:);
        tXTrTr = tXTr(idxTr,:);
        
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


