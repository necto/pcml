function [ Prediction, Confidence ] = NNPredict( test )
addpath(genpath('./DeepLearnToolbox'))
%NNPREDICT Summary of this function goes here
%   Detailed explanation goes here
  S = load('models/NeuralNetwork');
  nn = S.nn;
  mu = S.mu;
  sigma = S.sigma;
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

  [max1, i] = max(nnPred,[], 2);
  for l = 1:size(nnPred,1);
    nnPred(l, i(l)) = -inf;
  end
  secondMax = max(nnPred,[], 2);
  Confidence = max1 .* (max1 - secondMax);
end

