clearvars;

% -- GETTING STARTED WITH THE IMAGE CLASSIFICATION DATASET -- %
% IMPORTANT:
%    Make sure you downloaded the file train.tar.gz provided to you
%    and uncompressed it in the same folder as this file resides.

% Load features and labels of training data
load train/train.mat;
% load test.mat;
addpath(genpath('./piotr_toolbox'));



%% split randomly into train/test, use K-fold
fprintf('Splitting into train/test..\n');

K = 5;
N = size(train.y, 1);
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K, Nk);
for k = 1:K
    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
end;

%% Cross validation
BERSub = zeros(K, 1);
for k = 1:K
  [Tr, Te] = split4crossValidation(k, idxCV, train);
  BERSub(k) = NeuralNetwork(Tr, Te); 
end
ber = mean(BERSub);

pause;
% save('pred_binary', 'Ytest');

%% Visualize wrong predictions
figure;
for i = 1:size(classVote, 1)
  if (classVote(i) ~= Te.y(i))
    clf();
    img = imread( sprintf('train/imgs/train%05d.jpg', Te.idxs(i)) );
    imshow(img);
    % show if it is classified as pos or neg, and true label
    title(sprintf('Label: %d, Pred: %d', train.y(Te.idxs(i)), classVote(i)));

    % pause;
  end
end