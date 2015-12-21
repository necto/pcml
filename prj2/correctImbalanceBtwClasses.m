function [ balancedTrainingSet ] = correctImbalanceBtwClasses( train )
%% Correct imbalance between classes.
% There are 964 airplanes, 1162 cars, 1492 horses and 2382 others objects.
% Here we select randomly 964 objects of each class to create a balaced
% training set

  minNbr = 964;
  % Pick randomly 964 samples per class
  idx1 = find(train.y==1);
  perm = randperm(length(idx1));
  idx1 = idx1(perm);
  idx2 = find(train.y==2);
  perm = randperm(length(idx2));
  idx2 = idx2(perm);
  idx2 = idx2(1:minNbr);
  idx3 = find(train.y==3);
  perm = randperm(length(idx3));
  idx3 = idx3(perm);
  idx3 = idx3(1:minNbr);
  idx4 = find(train.y==4);
  perm = randperm(length(idx4));
  idx4 = idx4(perm);
  idx4 = idx4(1:minNbr);

  % Create unique idxTr to copy samples from training set "train"
  idxTr = vertcat(idx1, idx2, idx3, idx4);
  % Shuffle the samples so there is not all samples from class 1 followed 
  % by all of class 2...
  perm = randperm(length(idxTr));
  idxTr = idxTr(perm);

  % Pack selected data (hog, cnn, y) into training set
  balancedTrainingSet.X_hog = train.X_hog(idxTr,:);
  balancedTrainingSet.X_cnn = train.X_cnn(idxTr,:);
  balancedTrainingSet.y = train.y(idxTr);
end

