function [ L, g ] = penLogisticRegLoss(beta, tX, y, lambda)
    % Compute the sigmoid (1/(1 + exp(-Xt*beta)) value;
    txb = tX*beta;
    sigma = sigmoid(txb);
    L = - y'*txb + ones(1, size(tX, 1))*log(1 + exp(txb)) + lambda*(beta'*beta);
    g = tX'*(sigma - y) + 2*lambda*beta;
    %S = diag(sigma)*diag(1 - sigma);
    %Id = eye(size(tX,2));Id(1,1)=0;
    %H = tX'*S*tX + lambda*Id;
end

