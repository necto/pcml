function [ y_pred, ber ] = RandomForest( Tr, Te, prior )
% Classificatio using a random forest
  NumTrees = 300;
  B = TreeBagger(NumTrees, Tr.X_cnn, Tr.y, 'OOBPred','On');
  y_pred = str2num(cell2mat(predict(B, Te.X_cnn)));
  ber = BERM(Te.y, y_pred);
end