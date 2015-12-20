function [ Prediction, Confidence ] = SVMPredict( data )
%SVMPREDICT Summary of this function goes here
%   Detailed explanation goes here

    svmC1 = load('models/svmC1.mat', 'SVMModel');
    svmC1 = svmC1.('SVMModel');
    svmC2 = load('models/svmC2.mat', 'SVMModel');
    svmC2 = svmC2.('SVMModel');
    svmC3 = load('models/svmC3.mat', 'SVMModel');
    svmC3 = svmC3.('SVMModel');
    
    HOG = zscore(data.X_hog);
    
    [p1, c1] = predict(svmC1, HOG);
    [p2, c2] = predict(svmC2, HOG);
    [p3, c3] = predict(svmC3, HOG);
    
    c1 = c1(:,1)/max(c1(:,1));
    c2 = c2(:,1)/max(c2(:,1));
    c3 = c3(:,1)/max(c3(:,1));
    
    ConfidenceAll = [c1 c2 c3];
    PredictionAll = [p1 p2 p3];
    
    Prediction = zeros(size(data, 1), 1) + 4; % Class 'other' by default

    [Confidence, Idx] = max(ConfidenceAll, [], 2);
    positiveClass = Idx;
    negativeClass = (PredictionAll(Idx) == -1);
    
    Prediction(negativeClass) = 4;
    Prediction(~negativeClass) = positiveClass(~negativeClass);
    Prediction = Prediction';
end
