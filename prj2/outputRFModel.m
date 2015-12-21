function outputRFModel( train )
% This function create an ensemble of decision trees for predicting label 
% as function of the CNN features. The random forest is saved to the file
% "models/RandomForest.mat".

  % The random forest is sensitive to imbalance between classes. In our
  % training set: there are 964 airplanes, 1162 cars, 1492 horses and 2382 
  % others objects.
  % We select randomly 964 objects of each class to create a balaced
  % training set
  train = correctImbalanceBtwClasses(train);

  % Create and save the decision trees in "models/RandomForest.mat"
  NumTrees = 300;
  B = TreeBagger(NumTrees, train.X_cnn, train.y, 'OOBPred','On','NumPrint',10);
  save('models/RandomForest', 'B');
end