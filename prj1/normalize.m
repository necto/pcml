function [ X, X_mean, X_std ] = normalize( X )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    X_mean = mean(X);
    X_std = std(X);
    X = adjust(X, X_mean, X_std);
end
