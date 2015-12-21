function [ Prediction, Confidence ] = EnsemblePredict( data )
% Predict lables using a combination of SVM, NN and Random Forests

% Run all three basic predictors.
[SVMPrediction, SVMConfidence] = SVMPredict(data);
[NNPrediction, NNConfidence] = NNPredict(data);
[RFPrediction, RFConfidence] = RandomForestPredict(data);

PredictionAll = [SVMPrediction NNPrediction RFPrediction];
ConfidenceAll = [SVMConfidence NNConfidence RFConfidence];

% No predictions, and no confidence by default.
Prediction = zeros(size(PredictionAll, 1), 1);
Confidence = zeros(size(ConfidenceAll, 1), 1);

% Counters for statistics.
byMajority = 0;
byConfidence = 0;

% Compute each prediciton individually
for i = 1:size(PredictionAll, 1)
    predictions = PredictionAll(i,:);
    % Find out the most frequent predicted class.
    [pred, freq] = mode(predictions);
    hasMajority = any(freq > 1);
    % Prefer the major predictoin,
    if (hasMajority)
        Prediction(i) = pred;
        Confidence(i) = max(ConfidenceAll(i, predictions == pred));
        byMajority = byMajority + 1;
    else % if all the predictors disagree, choose the most confident one.
        [Confidence(i), idConf] = max(ConfidenceAll(i,:));
        Prediction(i) = PredictionAll(i, idConf);
        byConfidence = byConfidence + 1;
    end
end

fprintf('by majority: %d; by confidence: %d\n', byMajority, byConfidence);
end
