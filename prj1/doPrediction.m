function [ y ] = doPrediction( X, beta1, beta2, beta3, th1, th2 )
    
    % Do predictions
    idx1 = find(X(:,58)<=th1);
    idx2 = find(X(:,58)>th1 & X(:,43)<th2);
    idx3 = find(X(:,43)>=th2);
    y(idx1) = X(idx1,:)*beta1;
    y(idx2) = X(idx2,:)*beta2;
    y(idx3) = X(idx3,:)*beta3;
    y = y';
end