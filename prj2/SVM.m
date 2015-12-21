function [ y_pred, ber ] = SVM( Tr, Te )
%% Binary classification with SVM and CNN features
  % y should be binary: either 2 for class {Airplane(1), Car(2), Horses(3)}
  % or 4 for Other
  y = Tr.y;
  y(Tr.y < 4) = 2;
  y(Tr.y == 4) = 1;
  
  % Train binary support vector machine classifier
  SVMModel = fitcsvm(Tr.X_cnn, y, 'KernelFunction','linear','Standardize',...
  true,'ClassNames',{'1','2'});

  % Predict the label for the testing set Te and compute BER
  [label,score] = predict(SVMModel,Te.X_cnn);
  y_pred = str2num(cell2mat(label));
  ber = BER(Te.y, y_pred, 2);
end

