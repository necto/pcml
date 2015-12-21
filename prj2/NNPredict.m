function [ Prediction, Confidence ] = NNPredict( test )
% A Neural Network has been trained and saved in "models/NeuralNetwork.mat".
% Here we simply load it and predict label and confidence for the testing 
% set "test".

  addpath(genpath('./DeepLearnToolbox'));
  % Load Neural Network and mu, sigma
  S = load('models/NeuralNetwork');
  nn = S.nn;
  mu = S.mu;
  sigma = S.sigma;  % We do that otherwise there is an error with matlab. 
                    % It use a function sigma instead of variable sigma
  test.normX = normalize(test.X_cnn, mu, sigma);  % normalize test data

  % to get the scores we need to do nnff (feed-forward)
  %  see for example nnpredict().
  % (This is a weird thing of this toolbox)
  nn.testing = 1;
  nn = nnff(nn, test.normX, zeros(size(test.normX,1), nn.size(end)));
  nn.testing = 0;

  % predict on the test set
  nnPred = nn.a{end};

  % get the most likely class
  [~,Prediction] = max(nnPred,[],2);

  % For each sample, scores contains a score for each class
  % Compute confidence as highestScore*(highestScore-secondHighestScore)
  [max1, i] = max(nnPred,[], 2);
  for l = 1:size(nnPred,1);
    nnPred(l, i(l)) = -inf;
  end
  secondMax = max(nnPred,[], 2);
  Confidence = max1 .* (max1 - secondMax);
end

