function [ L, g ] = penLogisticRegLoss(beta, tX, y, lambda)
    % Compute the sigmoid (1/(1 + exp(-Xt*beta)) value;
    sigma = sigmoid(tX*beta);
    L = - y'*tX*beta + ones(1, length(tX))*log(1 + exp(tX*beta));
    g = tX'*(sigma - y);
    S = diag(sigma)*diag(1 - sigma);
    Id = eye(size(tX,2));Id(1,1)=0;
    H = tX'*S*tX + lambda*Id;
end

