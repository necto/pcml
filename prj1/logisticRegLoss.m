% Compute two fit characteristics for the given beta in the regression
% problem defined by (tX, y);
% Returns a vector of two values:
%  L - the loss function - a logistic error, approximating how far the beta
%      are from the desired solution.
%  g - a gradient vector for L, giving an idea on the direction of the
%      function growth.
function [ L, g ] = logisticRegLoss( beta, tX, y )
  % Compute the sigmoid (1/(1 + exp(-Xt*beta)) value;
  sigma = sigmoid(tX*beta);
  L = - y'*tX*beta + ones(1, length(tX))*log(1 + exp(tX*beta));
  g = tX'*(sigma - y);
end

