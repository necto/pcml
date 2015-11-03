% Penalized logistic regression using gradient descent with step size alpha
% and regularization parameter lambda
function [ beta ] = penLogisticRegressionEx(y, tX, alpha, lambda, epsilon)
  % Set the initial beta
  beta = zeros(size(tX,2), 1);%ones(size(tX,2), 1)*1e-3;
  
  % Stop when the difference between two L is less than epsilon or 
  % after maxIters iterations
  maxIters = 500000;
  
  % Keeps whether the last iteration was slowing down.
  tendence = false;
  for k = 1:maxIters
      [L, g] = penLogisticRegLoss(beta, tX, y, lambda);
      beta = beta - alpha * g;
      % Check convergence
      if(k > 1)
          delta = L_prev - L;
          % Check if the progress has slown down, meaning that we are close
          % to a local optimum.
          if(abs(delta)/L < epsilon)
              if (tendence) % Make sure it is not by a chance
                  % Check that we iterated enough, and not just happend to
                  % start at an optimum.
                  if (k < 4)
                      disp('warning: penLogisticRegresssion converged too fast');
                  end;
                  return;
              else
                  tendence = true; % If this situation repeats, we can stop.
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
  % even maxIters was not enough to satisfy the epsilon-condition. Probably
  % you should enlarge alpha, or epsilon.
  fprintf('penalized logistic regression did not converge (%d;%d)\n', L, delta);
end

