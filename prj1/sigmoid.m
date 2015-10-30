% Return the sigmoid of x given by the following equation:
% f(x) = 1 / (1 + e^(-x))
function [ sig ] = sigmoid( x )
    sig(x >= 0) = 1 ./ (1 + exp(-x(x>=0)));
    sig(x < 0) = exp(x(x<0)) ./ (1 + exp(x(x<0)));
    sig = sig';
end

