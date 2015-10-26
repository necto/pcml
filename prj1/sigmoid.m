%
function [ sigma ] = sigmoid( x )
    sigma = zeros(size(x));
    sigma = 1 ./ (1 + exp(-x));
end

