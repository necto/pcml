function [ y_pred, ber ] = RandomForest( Tr, Te )
% Do multiclass classification using a random forest of 300 trees
  NumTrees = 300; 
  % Create ensemble of decision trees for predicting label as function of
  % the CNN features
  B = TreeBagger(NumTrees, Tr.X_cnn, Tr.y, 'OOBPred','On', 'NumPrint',10);

  % Predict the label for the testing set Te and compute BER 
  y_pred = str2num(cell2mat(predict(B, Te.X_cnn)));
  ber = BERM(Te.y, y_pred);
end