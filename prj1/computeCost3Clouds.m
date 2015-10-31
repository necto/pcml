function [ L ] = computeCost3Clouds( y, tX, idx1, idx2, idx3, beta1, beta2, beta3 )

    % Do predictions
    pred = zeros(size(y));
    pred(idx1) = tX(idx1,:)*beta1;
    pred(idx2) = tX(idx2,:)*beta2;
    pred(idx3) = tX(idx3,:)*beta3;

    N = length(y);
    %compute error
    e = y - pred;
    %compute MSE
    L = e'*e/(2*N);
end

