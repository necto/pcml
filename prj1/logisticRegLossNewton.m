% Compute three fit characteristics for the given beta in the regression
% problem defined by (tX, y), for a Newton logistic regression algorithm;
% Returns a vector of three values:
%  L - the loss function - a logistic error, approximating how far the beta
%      are from the desired solution.
%  g - a gradient vector for L, giving an idea on the direction of the
%      function growth.
%  H - the Hessian matrix.
function [ L, g, H] = logisticRegLossNewton( beta, tX, y )
  % Compute the sigmoid (1/(1 + exp(-Xt*beta)) value;
  sigma = sigmf(tX*beta, [1 0]);
  L = - y'*tX*beta + ones(1, length(tX))*log(1 + exp(tX*beta));
  g = tX'*(sigma - y);
  S = diag(sigma)*diag(1 - sigma);
  H = tX'*S*tX;
end

