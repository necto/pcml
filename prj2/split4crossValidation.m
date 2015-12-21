function [ Tr, Te ] = split4crossValidation( k, idxCV, train )
%   K-Fold: Split data into train and validation test
%   The function returns two structure (Tr, Te) containing the indexes, 
%   HOG feature, CNN feature and label y
  Tr = [];
  Te = [];
  
  % Compute K-fold idx for training and testing sets
  idxTe = idxCV(k, :);
  idxTe = idxTe(:);
  idxTr = idxCV([1:k-1 k+1:end],:);
  idxTr = idxTr(:); 
  
  % Pack data (idx, hog, cnn, y) into training set
  Tr.idxs = idxTr;
  Tr.X_hog = train.X_hog(Tr.idxs,:);
  Tr.X_cnn = train.X_cnn(Tr.idxs,:);
  Tr.y = train.y(Tr.idxs);

  % Pack data (idx, hog, cnn, y) into testing set
  Te.idxs = idxTe;
  Te.X_hog = train.X_hog(Te.idxs,:);
  Te.X_cnn = train.X_cnn(Te.idxs,:);
  Te.y = train.y(Te.idxs);
end