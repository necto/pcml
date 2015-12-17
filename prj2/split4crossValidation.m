function [ Tr, Te ] = split4crossValidation( k, idxCV, train )
%   K-Fold: Split data into train and validation test
%   The function returns two structure (Tr, Te) containing the indexes, 
%   HOG feature, CNN feature and label y
  Tr = [];
  Te = [];
    
  idxTe = idxCV(k, :);
  idxTe = idxTe(:);
  idxTr = idxCV([1:k-1 k+1:end],:);
  idxTr = idxTr(:); 
  
  Tr.idxs = idxTr;
  Tr.X_hog = train.X_hog(Tr.idxs,:);
  Tr.X_cnn = train.X_cnn(Tr.idxs,:);
  Tr.y = train.y(Tr.idxs);

  Te.idxs = idxTe;
  Te.X_hog = train.X_hog(Te.idxs,:);
  Te.X_cnn = train.X_cnn(Te.idxs,:);
  Te.y = train.y(Te.idxs);
end