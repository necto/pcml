% Linear regression loss function.
function [ L ] = computeCost( y, tX, beta )
    N = length(y);
    %compute error
    e = y - tX*beta;
    %compute MSE
    L = e'*e/(2*N);
end

