% Split the data into train(XTr, yTr) and validation(XTe, yTe) sets.
function [ yTe, yTr, XTe, XTr ] = split4crossValidation( k, idxCV, y, X )
    idxTe = idxCV(k, :);
    idxTe = idxTe(:);
    idxTr = idxCV([1:k-1 k+1:end],:);
    idxTr = idxTr(:);
    yTe = y(idxTe);
    yTr = y(idxTr);
    XTe = X(idxTe,:);
    XTr = X(idxTr,:);
end