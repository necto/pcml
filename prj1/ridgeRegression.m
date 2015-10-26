% Ridge regression using normal equations with regularization coefficient 
% lambda.
function [ beta ] = ridgeRegression(y, tX, lambda)
  Id = lambda * eye(size(tX,2));
  Id(1,1) = 0;
  beta = inv(tX'*tX+Id)*tX'*y;
end

