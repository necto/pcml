clearvars;
close all;
% Load features and labels of training data
load train/train.mat;
% Load features of testing data
% load test.mat;
addpath(genpath('./piotr_toolbox'));

%% Set which method to run
multiclassNN = false;     % Neural network with 4 classes
binaryNN = false;          % Neural network with 2 classes
randomForest = true;     % Random forest
showWrongPred = false;    % Shows images with wrong prediction
svm = false;

%% split randomly into train/test, use K-fold
fprintf('Splitting into train/test..\n');
K = 3;
N = size(train.y, 1)/10;
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K, Nk);
for k = 1:K
    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
end;

%% Neural network with 2 classes
if(binaryNN)
  BERSub = zeros(K, 1);
  predSub = cell(K, 1);
  for k = 1:K
    [Tr, Te] = split4crossValidation(k, idxCV, train);
    [predSub{k}, BERSub(k)] = NeuralNetworkBinary(Tr, Te); 
  end
  ber = mean(BERSub);
  fprintf('\nK-fold(K = %d) BER for binary NN: %.2f%%\n\n', K, 100*ber ); 
end;

%% Neural network with 4 classes
if(multiclassNN)
  BERSub = zeros(K, 1);
  predSub = cell(K, 1);
  for k = 1:K
    [Tr, Te] = split4crossValidation(k, idxCV, train);
    [predSub{k}, BERSub(k)] = NeuralNetwork(Tr, Te); 
  end
  ber = mean(BERSub);
  fprintf('\nK-fold(K = %d) BER for Multiclass NN: %.2f%%\n\n', K, 100*ber ); 
end;


%% Random forest
if(randomForest)
  fprintf('\nRandom Forest\n'); 
  BERSub = zeros(K, 1);
  predSub = cell(K, 1);
  for k = 1:K
    [Tr, Te] = split4crossValidation(k, idxCV, train);
    [predSub{k}, BERSub(k)] = RandomForest(Tr, Te); 
  end
  ber = mean(BERSub);
  fprintf('\nK-fold(K = %d) BER for Random forest: %.2f%%\n\n', K, 100*ber ); 
end

%% Support vector machine
if(svm)
  fprintf('\nSVM\n'); 
  BERSub = zeros(K, 1);
  predSub = cell(K, 1);
  for k = 1:K
    [Tr, Te] = split4crossValidation(k, idxCV, train);
    [predSub{k}, BERSub(k)] = SVM(Tr, Te); 
  end
  ber = mean(BERSub);
  fprintf('\nK-fold(K = %d) BER for SVM: %.2f%%\n\n', K, 100*ber ); 
end


%% Save results
% save('pred_binary', 'Ytest');


%% Visualize wrong predictions
if(showWrongPred)
  figure;
  for i = 1:size(classVote, 1)
    if (classVote(i) ~= Te.y(i))
      clf();
      img = imread( sprintf('train/imgs/train%05d.jpg', Te.idxs(i)) );
      imshow(img);
      title(sprintf('Label: %d, Pred: %d', train.y(Te.idxs(i)), classVote(i)));
    end
  end
end