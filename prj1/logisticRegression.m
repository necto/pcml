% A logistic regression classification algorithm through a gradient
% descent. Iteratively approximates the binary y with 
% (0.5 < sigmoid(-tX*beta)) moving beta by alpha*gradient towards the
% minimum of the logistic function.
% Returns the beta that approximates y good enough if a reasonable amount
% of iterations is enough to reach it.
function [ beta ] = logisticRegression( y, tX, alpha )
  % Set the start values to begin with.
  beta = ones(size(tX,2), 1)*1e-3;
  
  % Termination metrics. The loop terminates either when two consequent
  % L values are closer than epsilon, or after maxIters iterations,
  % whicever happens earlier.
  maxIters = 1000000;
  epsilon = 1e-5;
  
  for k = 1:maxIters
     [L, g] = logisticRegLoss(beta, tX, y);
     beta = beta - alpha*g;
     
     if (k > 1)
         % Check for the termination condition: L changed insignificantly.
         converged = abs(L_prev - L) < epsilon;
         if(converged)
             break;
         end;
     end;
     L_prev = L; % Remember the last iteration L for the termination check.
  end;
  
end

