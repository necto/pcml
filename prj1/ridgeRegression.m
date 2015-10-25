% Ridge regression using normal equations with regularizytion coefficient 
% lambda.
function [ beta ] = ridgeRegression(y, tX, lambda)
    beta0 = [-100:1:200];
    L = zeros(length(beta0));
    N = length(y)
    for i = 1:length(beta0)
          L(i) = computeCost(y, tX, beta0(i)) + lambda/(2*N)*beta0(i)^2;
    end

end

