% Compute the least squares solution in a single step. Requires that the
% Gram matrix tX'*tX is invertible, i.e. tX kernel is empty.
% Returns the beta - solution for the given regression problem (y, tX).
function [ beta ] = leastSquares( y, tX )
  if (rcond(tX'*tX) < 1e-5) % Check that the Gram matrix is invertible.
      disp('warning the X matrix is ill-conditioned');
  end;
  beta = (tX'*tX)\(tX'*y);
end

