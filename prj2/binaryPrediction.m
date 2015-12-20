clearvars;
close all;
% Load features and labels of training data
load train/train.mat;
% Load features of testing data
% load test.mat;
addpath(genpath('./piotr_toolbox'));

% Classes
Airplane = 1;
Car = 2;
Horse = 3;
Other = 4;

%% Set which method to run
multiclassNN = false;     % Neural network with 4 classes
binaryNN = false;          % Neural network with 2 classes
randomForest = true;     % Random forest
showWrongPred = false;    % Shows images with wrong prediction
svm = false;
rF = false;

%% split randomly into train/test, use K-fold
fprintf('Splitting into train/test..\n');
K = 3;
N = size(train.y, 1);
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

if(rF)
  fprintf('\nRandom Forest combination\n'); 
  BERSub = zeros(K, 1);
  predSub = cell(K, 1);
  predSubPlane = cell(K, 1);
  predSubCar = cell(K, 1);
  predSubHorse = cell(K, 1);
  predSubOther = cell(K, 1);
  predCombination = cell(K, 1);
  for k = 1:K
    [Tr, Te] = split4crossValidation(k, idxCV, train);
    [predSub{k}, ~] = RandomForest(Tr, Te); 
    % Airplane
    TrPlane = Tr;
    TrPlane.y(TrPlane.y == Airplane) = 2;
    TrPlane.y(TrPlane.y ~= Airplane) = 1;
    [predSubPlane{k}, ~] = RandomForest(TrPlane, Te);
    % Car
    TrCar = Tr;
    TrCar.y(TrCar.y == Car) = 2;
    TrCar.y(TrCar.y ~= Car) = 1;
    [predSubCar{k}, ~] = RandomForest(TrCar, Te); 
    % Horse
    TrHorse = Tr;
    TrHorse.y(TrHorse.y == Horse) = 2;
    TrHorse.y(TrHorse.y ~= Horse) = 1;
    [predSubHorse{k}, ~] = RandomForest(TrHorse, Te); 
    % Other
    TrOther = Tr;
    TrOther.y(TrOther.y == Other) = 2;
    TrOther.y(TrOther.y ~= Other) = 1;
    [predSubOther{k}, ~] = RandomForest(TrOther, Te); 
    
    yPred = predSub{k};
    yPred(predSubPlane{k} == 2) = Airplane;
    yPred(predSubCar{k} == 2) = Car;
    yPred(predSubHorse{k} == 2) = Horse;
    yPred(predSubOther{k} == 2) = Other;
    
    predCombination{k} = yPred;
    BERSub(k) = BER(Te.y, yPred, 4);
  end
  ber = mean(BERSub);
  fprintf('\nK-fold(K = %d) BER for Random forest: %.2f%%\n\n', K, 100*ber ); 
  
end

%% Save results
% save('pred_binary', 'Ytest');
