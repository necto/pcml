% Return the sigmoid of x given by the following equation:
% f(x) = 1 / (1 + e^(-x))
function [ sig ] = sigmoid( x )
    sig = zeros(size(x));
    sig = 1 ./ (1 + exp(-x));
end

