function [ Prediction, Confidence ] = EnsemblePredict( train )
%EnsemblePredict Summary of this function goes here
%   Detailed explanation goes here

[SVMPrediction, SVMConfidence] = SVMPredict(train);
[NNPrediction, NNConfidence] = NNPredict(train);
[RFPrediction, RFConfidence] = RandomForestPredict(train);

PredictionAll = [SVMPrediction NNPrediction RFPrediction];
ConfidenceAll = zscore([SVMConfidence NNConfidence RFConfidence]);

[Confidence, ConfidentIdx] = max(ConfidenceAll, [], 2);

Prediction = PredictionAll(sub2ind(size(PredictionAll), ...
                           1:length(ConfidenceIdx), ConfidenceIdx'));
end

                       