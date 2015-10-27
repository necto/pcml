 function Xpoly = myPoly(X,degree)
% build matrix Phi for polynomial regression of a given degree

  Xpoly = X;
  XpolyPrev = Xpoly;
  for k = 1:(degree-1)
    Xpoly = zeros(size(X,1), size(XpolyPrev, 2)*size(X,2));
    for i = 1:size(X,1)
        poly = XpolyPrev(i,:)'*X(i,:);
        Xpoly(i,:) = poly(:);
    end
    XpolyPrev = Xpoly;
    %for i = 1:size(X,2)
    %   Xpoly(:,k*i) = X(:,i).^k;
    %end
  end
end

