function [ Prediction, Confidence ] = SVMPredict( data )
%SVMPREDICT Summary of this function goes here
%   Detailed explanation goes here

    svmC1 = load('models/svmC1.mat', 'SVMModel');
    svmC1 = svmC1.('SVMModel');
    svmC2 = load('models/svmC2.mat', 'SVMModel');
    svmC2 = svmC2.('SVMModel');
    svmC3 = load('models/svmC3.mat', 'SVMModel');
    svmC3 = svmC3.('SVMModel');
    
    [p1, c1] = predict(svmC1, data);
    [p2, c2] = predict(svmC2, data);
    [p3, c3] = predict(svmC3, data);
    
    ConfidenceAll = [c1 c2 c3];
    
    Prediction = zeros(size(data, 1), 1) + 4; % Class 'other' by default

    [Confidence, Idx] = max(ConfidenceAll, [], 2);
    negativeClass = (rem(Idx, 2) == 0);
    positiveClass = Idx/2;
    
    Prediction(negativeClass) = 4;
    Prediction(~negativeClass) = positiveClass(~negativeClass);
end
