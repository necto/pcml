 function Xpoly = myPoly(X,degree)
% build matrix Phi for polynomial regression of a given degree

  for k = 1:degree
    for i = 1:size(X,2)
       Xpoly(:,k*i) = X(:,i).^k;
    end
  end
end

