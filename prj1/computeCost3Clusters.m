% Linear regression loss function.
function [ L ] = computeCost3Clusters( y, tX, idxTr, b1, b2, b3 )
    N = length(y);
    e = zeros(size(y,1),1);
    %compute error
    e(idxTr==1) = y(idxTr==1) - tX(idxTr==1,:)*b1;
    e(idxTr==2) = y(idxTr==2) - tX(idxTr==2,:)*b2;
    e(idxTr==3) = y(idxTr==3) - tX(idxTr==3,:)*b3;

    %compute MSE
    L = e'*e/(2*N);
end

