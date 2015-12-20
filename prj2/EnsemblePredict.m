function [ Prediction, Confidence ] = EnsemblePredict( data )
%EnsemblePredict Summary of this function goes here
%   Detailed explanation goes here

[SVMPrediction, SVMConfidence] = SVMPredict(data);
[NNPrediction, NNConfidence] = NNPredict(data);
[RFPrediction, RFConfidence] = RandomForestPredict(data);

PredictionAll = [SVMPrediction NNPrediction RFPrediction];
ConfidenceAll = [SVMConfidence NNConfidence RFConfidence];
[ConfidenceAllNorm, ~, ~] = zscore(ConfidenceAll);

[~, ConfidenceIdx] = max(ConfidenceAllNorm, [], 2);

fprintf('used %d SVM, %d NN and %d RF predictions\n', ...
        length(find(ConfidenceIdx == 1)), ...
        length(find(ConfidenceIdx == 2)), ...
        length(find(ConfidenceIdx == 3)));

Prediction = PredictionAll(sub2ind(size(PredictionAll), ...
                           1:length(ConfidenceIdx), ConfidenceIdx'));
Confidence = ConfidenceAll(sub2ind(size(ConfidenceAll), ...
                           1:length(ConfidenceIdx), ConfidenceIdx'));
end
