clear all;
load('data/classification.mat');

tX_train = [ones(size(X_train, 1), 1) X_train];
y_train = (y_train + 1)/2; % !!! remember invert that for predictions

[tXTr, yTr, tXTe, yTe] = split(y_train, tX_train, 0.7);

[tXTr, XTr_mean, XTr_std] = normalize(tXTr);

tXTe = adjust(tXTe, XTr_mean, XTr_std);

lrBeta = logisticRegression(yTr, tXTr, 0.001);


lrY = sigmoid(-tXTe * lrBeta) > 0.5;

K = 2;
N = size(yTr, 1);
idx = randperm(N);
Nk = floor(N/K);
for k = 1:K
    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
end;

lvals = logspace(-2, 2, 5);

for l = 1:length(lvals)
    lambda = lvals(l);
    
    for k = 1:K
        idxTe = idxCV(k, :);
        idxTr = idxCV([1:k-1 k+1:end]);
        idxTr = idxTr(:);
        yTrTe = yTr(idxTe);
        yTrTr = yTr(idxTr);
        tXTrTe = tXTr(idxTe,:);
        tXTrTr = tXTr(idxTr,:);
        
        beta = penLogisticRegression(yTrTr, tXTrTr, 0.003, lambda);
        
        errorTeSub(k) = (sum((sigmoid(-tXTrTe*beta) > 0.5) ~= yTrTe))/size(yTrTe,1);
    end;
    errorTe(l) = mean(errorTeSub);
end;
[errStar lStar] = min(errorTe);



