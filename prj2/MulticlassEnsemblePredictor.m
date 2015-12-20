clearvars;
close all;
load train/train.mat;

[SVMPrediction, SVMConfidence] = SVMPredict(train);
[NNPrediction, NNConfidence] = NNPredict(train);
[RFPrediction, RFConfidence] = RandomForestPredict(train);

PredictionAll = [SVMPrediction NNPrediction RFPrediction];
ConfidenceAll = zscore([SVMConfidence NNConfidence RFConfidence]);

[Confidence, ConfidentIdx] = max(ConfidenceAll, [], 2);

Prediction = PredictionAll(sub2ind(size(PredictionAll), ...
                           1:length(ConfidenceIdx), ConfidenceIdx'));

                       