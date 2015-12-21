clearvars;
close all;
% Load features and labels of training data
load train/train.mat;
% Load features of testing data
% load test.mat;
addpath(genpath('./piotr_toolbox'));

% Classes
[Airplane, Car, Horse, Other] = deal(1, 2, 3, 4);
train = correctImbalanceBtwClasses(train);
%% Set which method to run
multiclassNN  = false;	% Neural network with multiclassification
binaryNN      = false;	% Neural network with binary classification
randomForest  = true;	% Random forest
svm           = false;  % SVM
rF            = false;  % Combination of random forests
trainModels   = false;	% Train and save NN and RandomForest
testModels    = false;  % Load and test NN and RandomForest

%% Split randomly into train/test, use K-fold
fprintf('Splitting into train/test..\n');
K = 3;
N = size(train.y, 1);
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K, Nk);
for k = 1:K
    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
end;

%% Neural network for Binary classification
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

%% Neural network for Multiclass classification
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

%% Train random forests 1 vs others (i.e. Airplane vs. {Car, Horse, Others}
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

%% Create, train and save a NN and RandomForest into folder "models/"
if(trainModels)
  fprintf('\nOutput models\n');  
  outputNNModel(train);
  fprintf('\nNN saved\n');  
  outputRFModel(train);
  fprintf('\nRandom forest saved\n');
end

%% Load and test a NN and RandomForest from folder "models/"
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
  berNN = mean(BERSubNN);
  berRF = mean(BERSubRF);
  fprintf('\nK-fold(K = %d)\nBER for Random forest: %.2f%%\nBER for NN: %.2f%%\n',...
  K, 100*berRF, 100*berNN);
end
