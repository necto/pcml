function [ Prediction, Confidence ] = EnsemblePredict( data )
%EnsemblePredict Summary of this function goes here
%   Detailed explanation goes here

[SVMPrediction, SVMConfidence] = SVMPredict(data);
[NNPrediction, NNConfidence] = NNPredict(data);
[RFPrediction, RFConfidence] = RandomForestPredict(data);

PredictionAll = [SVMPrediction NNPrediction RFPrediction];
ConfidenceAll = [SVMConfidence NNConfidence RFConfidence];

Prediction = zeros(size(PredictionAll, 1), 1);
Confidence = zeros(size(ConfidenceAll, 1), 1);
classes = unique(PredictionAll);

byMajority = 0;
byConfidence = 0;

for i = 1:size(PredictionAll, 1)
    [pred freq] = mode(PredictionsAll(i));
    hasMajority = any(freq > 1);
    if (hasMajority)
        Prediction(i) = pred;
        Confidence(i) = max(ConfidenceAll(i, PredictionAll == pred));
        byMajority = byMajority + 1;
    else
        [Confidence(i), idConf] = max(ConfidenceAll(i));
        Prediction(i) = PredictionAll(idConf);
        byConfidence = byConfidence + 1;
    end
end

fprintf('by majority: %d; by confidence: %d\n', byMajority, byConfidence);
end
