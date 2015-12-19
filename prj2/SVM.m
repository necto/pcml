function [ y_pred, ber ] = SVM( Tr, Te )
  y = Tr.y;
  y(Tr.y < 4) = 2;
  y(Tr.y == 4) = 1;
  SVMModel = fitcsvm(Tr.X_cnn, y, 'KernelFunction','linear','Standardize',true,'ClassNames',{'1','2'});
  [label,score] = predict(SVMModel,Te.X_cnn);
  y_pred = str2num(cell2mat(label));
  ber = BER(Te.y, y_pred, 2);
end

