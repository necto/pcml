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

Prediction = PredictionAll(sub2ind(size(PredictionAll), ...
                           1:length(ConfidenceIdx), ConfidenceIdx'));
Confidence = ConfidenceAll(sub2ind(size(ConfidenceAll), ...
                           1:length(ConfidenceIdx), ConfidenceIdx'));
end
