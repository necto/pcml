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
  N = size(tX, 1);
  txb = tX*beta;
  L = - y'*txb/N + ones(1, N)*log1PlusExp(txb)/N;
  g = tX'*(sigma - y)/N;
end

