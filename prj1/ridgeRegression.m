% Ridge regression using normal equations with regularization coefficient 
% lambda.
function [ beta ] = ridgeRegression(y, tX, lambda)
  % Build matrix Id = Lambda * identity matrix ,with Id(1,1) = 0
  Id = lambda * eye(size(tX,2));
  Id(1,1) = 0;
  % Compute beta
  beta = inv(tX'*tX+Id)*tX'*y;
end

