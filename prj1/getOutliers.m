function [ outliers ] = getOutliers( X_train )
  outliers = zeros(size(X_train,1),1);
  outliers = outliers | (X_train(:,1)<1.5 | (X_train(:,1)>8));
  outliers = outliers | (X_train(:,10)<=-1.5);
  outliers = outliers | (X_train(:,13)>=4.5);
  outliers = outliers | (X_train(:,15)<1);
  outliers = outliers | (X_train(:,17)<-3 | (X_train(:,17)>3.7));
  outliers = outliers | (X_train(:,19)<=-0.5);
  outliers = outliers | (X_train(:,21)<1);
  outliers = outliers | (X_train(:,22)<-1.5);
  outliers = outliers | (X_train(:,24)<=-1.5);
  outliers = outliers | (X_train(:,25)>=7.5);
  outliers = outliers | (X_train(:,28)<=0.6);
  outliers = outliers | (X_train(:,30)<=-2.8);
  outliers = outliers | (X_train(:,32)<=1.8);
  outliers = outliers | (X_train(:,34)>=6.6);
  outliers = outliers | (X_train(:,36)>=7.8);
  outliers = outliers | (X_train(:,39)<0.5);
  outliers = outliers | (X_train(:,40)<=-0.6 | (X_train(:,40)>6.5));
  outliers = outliers | (X_train(:,41)<=-3);
  outliers = outliers | (X_train(:,44)<=1.5);
  outliers = outliers | (X_train(:,45)<=-2.7);
  outliers = outliers | (X_train(:,47)<=0);
  outliers = outliers | (X_train(:,49)>=4.2);
  outliers = outliers | (X_train(:,50)>=7.5);
  outliers = outliers | (X_train(:,54)<=-3.9 | (X_train(:,54)>=3.9));
  outliers = outliers | (X_train(:,55)<=0.6);
  outliers = outliers | (X_train(:,56)<=-1.5);
  outliers = outliers | (X_train(:,63)<=0 | (X_train(:,63)>=8.7));
  outliers = outliers | (X_train(:,64)<=-2.4);
end

