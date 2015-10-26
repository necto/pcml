% Ridge regression using normal equations with regularization coefficient 
% lambda.
function [ beta ] = ridgeRegression(y, tX, lambda)
  % Set the initial beta
  beta = ones(size(tX,2), 1)*1e-3;  
  phi = tX;
  Id = lambda * eye(size(tX,2));Id(1,1)=0;
  beta = inv(phi'*phi+lambda)*phi'*y;
end

