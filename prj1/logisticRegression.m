% A logistic regression classification algorithm through a gradient
% descent. Iteratively approximates the binary y with 
% (0.5 < sigmoid(-tX*beta)) moving beta by alpha*gradient towards the
% local minimum of the logistic function.
% Returns the beta that approximates y good enough if a reasonable amount
% of iterations is enough to reach it.
function [ beta ] = logisticRegression( y, tX, alpha)
    beta = logisticRegressionEx(y, tX, alpha, 1e-5);
end
