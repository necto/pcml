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
classes = unique(PredictionsAll);

byMajority = 0;
byConfidence = 0;

for i = 1:size(PredictionsAll, 1)
    hasMajority = any(histc(PredictionsAll(i), classes) > 1);
    if (hasMajority)
        Prediction(i) = mode(PredictionsAll(i));
        Confidence(i) = max(ConfidenceAll(i, PredictionAll == Prediction(i)));
        byMajority = byMajority + 1;
    else
        [Confidence(i), idConf] = max(ConfidenceAll(i));
        Prediction(i) = PredictionAll(idConf);
        byConfidence = byConfidence + 1;
    end
end

fprintf('by majority: %d; by confidency: %d\n', byMajority, byConfidence);
end
