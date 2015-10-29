function [ yTe, yTr, XTe, XTr ] = split4crossValidation( k, idxCV, y, X )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    idxTe = idxCV(k, :);
    idxTe = idxTe(:);
    idxTr = idxCV([1:k-1 k+1:end]);
    idxTr = idxTr(:);
    yTe = y(idxTe);
    yTr = y(idxTr);
    XTe = X(idxTe,:);
    XTr = X(idxTr,:);
end

