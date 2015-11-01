% Return the sigmoid of x given by the following equation:
% f(x) = 1 / (1 + e^(-x))
function [ sig ] = sigmoid( x )
    sig(x >= 0) = 1 ./ (1 + exp(-x(x>=0))); % Optimize for numerical
    sig(x < 0) = exp(x(x<0)) ./ (1 + exp(x(x<0))); % pecularities.
    sig = sig'; % for some reason x(cond) = y(cond) gives a row, not column.
end

