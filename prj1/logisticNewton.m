% A logistic regression classification algorithm through a Newton method.
% It iteratively approximates the binary y with 
% (0.5 < sigmoid(-tX*beta)) moving beta towards the minimum of an
% approximation paraboloid, computed by the Hessian of the logistic
% funcion. The alpha parameter adjust the size of the step in each
% iteration. Normally it should be substentially bigger than for the
% logisticRegression function.
% Returns the beta that approximates y good enough if a reasonable amount
% of iterations is enough to reach it.
function [ beta ] = logisticNewton( y, tX, alpha )
  % Initial values for beta to start with.
  beta = ones(size(tX,2), 1)*1e-3;

  % Termination parameters. The loop terminates whenever some of the events
  % happen: two subsequent logisitc values differ less than by epsilon, or
  % maxIters iterations performed.
  maxIters = 1000000;
  epsilon = 1e-5;
  
  for k = 1:maxIters
     [L, g, H] = logisticRegLossNewton(beta, tX, y);
     
     % Next, better beta values.
     beta = beta - alpha*(H\g);
     
     if (k > 1) % check if we are improving slowly - that would indicate
         converged = abs(L_prev - L) < epsilon;% that we are close to the
         if(converged) break; end;             % optimum.
     end;
     % Keep the previous L value for the termination condition in the next
     L_prev = L;% iteration.
  end;
end
