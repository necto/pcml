function [ Prediction, Confidence ] = SVMPredict( data )
% The One-vs-All multi class predictor based on binary SVM models.
    % Load the precomputed models. See Makefile for hos to get the files
    % svmC1.mat, svmC2.mat and svmC3.mat.
    svmC1 = load('models/svmC1.mat', 'SVMModel');
    svmC1 = svmC1.('SVMModel');
    svmC2 = load('models/svmC2.mat', 'SVMModel');
    svmC2 = svmC2.('SVMModel');
    svmC3 = load('models/svmC3.mat', 'SVMModel');
    svmC3 = svmC3.('SVMModel');
    
    % Normalize the input, because the models were trained on the
    % normalized features.
    HOG = zscore(data.X_hog);
    
    % Detect separately each class.
    [p1, c1] = predict(svmC1, HOG);
    [p2, c2] = predict(svmC2, HOG);
    [p3, c3] = predict(svmC3, HOG);
    
    % Compute the SVM confidence scores for each class.
    c1 = abs(c1(:,1)); c1 = c1/max(c1);
    c2 = abs(c2(:,1)); c2 = c2/max(c2);
    c3 = abs(c3(:,1)); c3 = c3/max(c3);
    
    ConfidenceAll = [c1 c2 c3];
    PredictionAll = [p1 p2 p3];
    
    % Prefer a object to the undetected one. Positive logic.
    PositivePrediction = (PredictionAll > 0);
    
    % Compute the confidence scores among the classifiers that are positive
    % that they see something.
    PositiveConfidence = zeros(size(ConfidenceAll,1), size(ConfidenceAll, 2));
    PositiveConfidence(PositivePrediction) = ConfidenceAll(PositivePrediction);
    
    Prediction = zeros(size(data, 1), 1) + 4; % Class 'other' by default

    [pConf, pIdx] = max(PositiveConfidence, [], 2);
    % Get the confidence for the negative predictions.
    [Confidence, ~] = min(ConfidenceAll, [], 2);
    
    % Predicate of wheter there is any of the detected classes
    hasPosPred = pConf > 0;
    
    Prediction(hasPosPred) = pIdx(hasPosPred);
    % The 'other' for inputs without positive results.
    Prediction(~hasPosPred) = 4;
    Confidence(hasPosPred) = pConf(hasPosPred);
    
    % Normalize the shape of the prediction.
    Prediction = Prediction';
end
