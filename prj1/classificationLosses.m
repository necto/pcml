% Compute the three required error metrics for the classification model
% beta and data (X, y).
function [ RMSE, zeroOne, logLoss ] = classificationLosses( X, beta, y )
    prediction = sigmoid(X*beta);
    N = length(y);
    RMSE = sqrt( (y - prediction)'*(y - prediction)/N);
    zeroOne = sum(y ~= (prediction > 0.5))/N;
    log0 = log(prediction);
    log1 = log(1 - prediction);
    log0(y == 0 & (log0 == NaN | isinf(log0))) = 0; % Delete singularity
    log1(y == 1 & (log1 == NaN | isinf(log1))) = 0; % points.
    logLoss = - (y'*log0 + (1-y)'*log1)/N;
end

