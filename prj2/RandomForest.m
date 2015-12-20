function [ y_pred, ber ] = RandomForest( Tr, Te )
% Classificatio using a random forest
  NumTrees = 300;
  B = TreeBagger(NumTrees, Tr.X_cnn, Tr.y, 'OOBPred','On');
  y_pred = str2num(cell2mat(predict(B, Te.X_cnn)));
  if(length(unique(Tr.y(:))) == 4)
    ber = BER(Te.y, y_pred, 4);
  else
    ber = BER(Te.y, y_pred, 2);
  end
end