function [ L, g ] = penLogisticRegLoss(beta, tX, y, lambda)
    % Compute the sigmoid (1/(1 + exp(-Xt*beta)) value;
    txb = tX*beta;
    sigma = sigmoid(txb);
    N = size(tX, 1);
    btail = beta(2:end);
    L = - y'*txb/N + ones(1, N)*log1PlusExp(txb)/N + lambda*(btail'*btail);
    g = tX'*(sigma - y)/N + 2*lambda*beta;
    %S = diag(sigma)*diag(1 - sigma);
    %Id = eye(size(tX,2));Id(1,1)=0;
    %H = tX'*S*tX + lambda*Id;
end

