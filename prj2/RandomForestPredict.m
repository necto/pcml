function [ Prediction, Confidence ] = RandomForestPredict( test )
%RANDOMFORESTPREDICT Summary of this function goes here
%   Detailed explanation goes here

  load('models/RandomForest.mat');
  [YFIT,scores] = predict(B, test.X_cnn);
  Prediction = str2num(cell2mat(YFIT));
  [max1, i] = max(scores,[], 2);
  for l = 1:size(scores,1);
    scores(l, i(l)) = -inf;
  end
  secondMax = max(scores,[], 2);
  Confidence = max1 .* (max1 - secondMax);
end


