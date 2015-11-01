% Loss function and its gradient for a penalized logistic regression model
% for the given (tX, y) data, lambda in the point beta.
% Returns L - the loss function value; g - the gradient.
function [ L, g ] = penLogisticRegLoss(beta, tX, y, lambda)
    % Compute the sigmoid (1/(1 + exp(-Xt*beta)) value;
    txb = tX*beta;
    sigma = sigmoid(txb);
    N = size(tX, 1);
    btail = beta(2:end);
    L = - y'*txb/N + ones(1, N)*log1PlusExp(txb)/N + lambda*(btail'*btail);
    g = tX'*(sigma - y)/N + 2*lambda*beta;
end

