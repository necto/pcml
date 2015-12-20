function outputRFModel( train )
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

  idxTr = vertcat(idx1, idx2, idx3, idx4);
  perm = randperm(length(idxTr));
  idxTr = idxTr(perm);

  train.X_hog = train.X_hog(idxTr,:);
  train.X_cnn = train.X_cnn(idxTr,:);
  train.y = train.y(idxTr); 
  NumTrees = 300;
  B = TreeBagger(NumTrees, train.X_cnn, train.y, 'OOBPred','On');
  save('models/RandomForest', 'B');
end