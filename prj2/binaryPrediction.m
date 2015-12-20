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
% Prior probabilities for each class
totalNbrTrainingSamples = length(train.y);
pA = sum(train.y==Airplane)/totalNbrTrainingSamples;
pC = sum(train.y==Car)/totalNbrTrainingSamples;
pH = sum(train.y==Horse)/totalNbrTrainingSamples;
pO = sum(train.y==Other)/totalNbrTrainingSamples;
prior = [pA, pC, pH, pO];

%% Set which method to run
multiclassNN = false;     % Neural network with 4 classes
binaryNN = false;          % Neural network with 2 classes
randomForest = false;     % Random forest
showWrongPred = false;    % Shows images with wrong prediction
svm = false;
rF = false;
trainModels = false;
testModels = true;
%% Correct imbalance between classes.
% There are 964 airplanes, 1162 cars, 1492 horses and 2382 others objects.
% Here we select randomly 964 objects of each class to create a balaced
% training set

% minNbr = 964;
% % Pick randomly 964 samples per class
% idx1 = find(train.y==1);
% perm = randperm(length(idx1));
% idx1 = idx1(perm);
% idx2 = find(train.y==2);
% perm = randperm(length(idx2));
% idx2 = idx2(perm);
% idx2 = idx2(1:minNbr);
% idx3 = find(train.y==3);
% perm = randperm(length(idx3));
% idx3 = idx3(perm);
% idx3 = idx3(1:minNbr);
% idx4 = find(train.y==4);
% perm = randperm(length(idx4));
% idx4 = idx4(perm);
% idx4 = idx4(1:minNbr);
% 
% idxTr = vertcat(idx1, idx2, idx3, idx4);
% perm = randperm(length(idxTr));
% idxTr = idxTr(perm);
% 
% train.X_hog = train.X_hog(idxTr,:);
% train.X_cnn = train.X_cnn(idxTr,:);
% train.y = train.y(idxTr);

% Build new training set


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
    [predSub{k}, BERSub(k)] = RandomForest(Tr, Te ); 
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

if(trainModels)
  fprintf('\nOutput models\n');  
  outputNNModel(train);
  %outputRFModel(train);
end

if(testModels)
  fprintf('\nTestModels\n'); 
  BERSubRF = zeros(K, 1);
  BERSubNN = zeros(K, 1);

  predSub = cell(K, 1);
  for k = 1:K
    [Tr, Te] = split4crossValidation(k, idxCV, train);
    [ PredictionNN, ConfidenceNN ] = NNPredict(Te); 
    BERSubNN(k) =  BERM( Te.y, PredictionNN );
    [ PredictionRF, ConfidenceRF ] = RandomForestPredict(Te); 
    BERSubRF(k) =  BERM( Te.y, PredictionRF );
  end
  berNN = mean(BERSubNN)
  berRF = mean(BERSubRF)
end
%% Save results
% save('pred_binary', 'Ytest');
