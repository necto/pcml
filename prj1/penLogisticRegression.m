% Penalized logistic regression using gradient descent with step size alpha
% and regularization parameter lambda
function [ beta ] = penLogisticRegression(y, tX, alpha, lambda, epsilon)
  % Set the initial beta
  beta = zeros(size(tX,2), 1);%ones(size(tX,2), 1)*1e-3;
  
  % Stop when the difference between two L is less than epsilon or 
  % after maxIters iterations
  maxIters = 500000;
  
  tendence = false;
  for k = 1:maxIters
      [L, g] = penLogisticRegLoss(beta, tX, y, lambda);
      beta = beta - alpha * g;
      % Check convergence
      if(k > 1)
          delta = L_prev - L;
          if(abs(delta)/L < epsilon)
              if (tendence)
                  if (k < 4)
                      disp('warning: penLogisticRegresssion converged too fast');
                  end;
                  return;
              else
                  tendence = true;
              end;
          else
              tendence = false;
          end;
          if (L == Inf)
              disp('Loss is infinite');
              break;
          end;
          if (L > L_prev)
              disp('warning: increasing loss');
          end;
      end;
      L_prev = L; % Remember the last iteration L for the termination check.
  end;
  fprintf('penalized logistic regression did not converge (%d;%d)\n', L, delta);
end

