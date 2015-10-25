% Penalized logistic regression using gradient descent with step size alpha
% and regularization parameter lambda
function [ beta ] = penLogisticRegression(y, tX, alpha, lambda)
    % Set the initial beta
    beta = ones(size(tX,2), 1)*1e-3;
  
    % Stop when the difference between two L is less than epsilon or 
    % after maxIters iterations
    maxIters = 10000;
    epsilon = 1e-5;
  
    for k = 1:maxIters
        [L, g] = penLogisticRegLoss(beta, tX, y);
        beta = beta - alpha * g;
        % Check convergence
        if(k > 1 && abs(L_prev - L) < epsilon) break; end; 
        L_prev = L; % Remember the last iteration L for the termination check.
    end;
end

