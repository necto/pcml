function [ Prediction, Confidence ] = RandomForestPredict( test )
% A random forest has been trained and saved in "RandomForest.mat".
% Here we simply load it and predict label and confidence for the testing 
% set "test".

  % Load decision trees
  load('models/RandomForest.mat');

  % Predict the label for the testing set test 
  [YFIT,scores] = predict(B, test.X_cnn);
  Prediction = str2num(cell2mat(YFIT));
  
  % For each sample, scores contains a score for each class
  % Compute confidence as highestScore*(highestScore-secondHighestScore)
  [max1, i] = max(scores,[], 2);
  for l = 1:size(scores,1);
    scores(l, i(l)) = -inf;
  end
  secondMax = max(scores,[], 2);
  Confidence = max1 .* (max1 - secondMax);
end


