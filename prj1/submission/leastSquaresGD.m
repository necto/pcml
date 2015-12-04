% Linear regression using gradient descent: Compute beta for the (y, Tx)
% problem and stepsize alpha
function [ beta ] = leastSquaresGD(y, tX, alpha)
  % Set initial beta
  beta = zeros(size(tX,2), 1);
  epsilon = 1e-9;
  % Check if the maximum number of iteration is reached
  maxItersReached = true;
  
  % Stop when g'*g is close to epsilon or after maxIters iterations
  maxIters = 1e6;  
  for k = 1:maxIters
     g = computeGradient(y, tX, beta);
     beta = beta - alpha * g;
     if g'*g < epsilon 
       maxItersReached = false;
       break; 
     end;
  end;
  % Display warning message if number max of iteration is reached
  if (maxItersReached)
      disp('Warning: Loop stopped because the max number of iteration was reached');
  end;
end

