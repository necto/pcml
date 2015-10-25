% Linear regression using gradient descent: Compute beta for the (y, Tx)
% problem and stepsize alpha
function [ beta ] = leastSquaresGD(y, tX, alpha)
  % Set initial beta
  beta = zeros(size(tX,2), 1);
  epsilon = 1e-5;
  
  % Stop when g'*g is close to epsilon or after maxIters iterations
  maxIters = 100000;  
  for k = 1:maxIters
     g = computeGradient(y, tX, beta);
     beta = beta - alpha * g;
     if g'*g < epsilon; break; end;
  end;
  
end

