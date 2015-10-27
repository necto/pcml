function [ X ] = adjust( X, X_mean, X_std )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    
    for k=1:size(X, 2)
        if (X_std(k) ~= 0)
            X(:,k) = (X(:,k) - X_mean(k))/X_std(k);
        else
            X(:,k) = X(:,k) - X_mean(k);
        end
    end

end

