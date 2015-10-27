load('data/classification.mat');

tX_train = [ones(size(X_train, 1), 1) X_train];
y_train = (y_train + 1)/2; % !!! remember invert that for predictions

%rng(0);
[tXTr, yTr, tXTe, yTe] = split(y_train, tX_train, 0.7);

[tXTr, XTr_mean, XTr_std] = normalize(tXTr);

tXTe = adjust(tXTe, XTr_mean, XTr_std);

beta = logisticRegression(yTr, tXTr, 0.001);


tY = sigmoid(-tXTe * beta) > 0.5;
assert(sum(tY ~= yTe) / size(yTe,1) < 0.2);

%K = 4;
%N = size(yTr);
%idx = randprem(N);

%Nk = floor(N/K);
%for k = 1:K
%    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
%end;


