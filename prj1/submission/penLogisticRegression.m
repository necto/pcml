% Penalized logistic regression using gradient descent with step size alpha
% and regularization parameter lambda
function [ beta ] = penLogisticRegression(y, tX, alpha, lambda)
    beta = penLogisticRegressionEx(y, tX, alpha, lambda, 1e-5);
end
