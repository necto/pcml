% Generate the multiclass and binary predicitons for the test data, and
% save it to the pred_multiclass.mat and pred_binary.mat as required by the
% project description.

load test/test.mat;

%% Multiclas prediction
fprintf('Performing multiclass prediction with the ensemble predictor.\n');
[Ytest, conf] = EnsemblePredict(test);

distr = histc(Ytest, [1 2 3 4]);
fprintf('Classifier found: %d Airplanes(1), %d Cars(2), %d Horses(3), %d Other(4).\n', ...
        distr(1), distr(2), distr(3), distr(4));

fprintf('Saving the predicted vector.\n');
save('pred_multiclass.mat', 'Ytest');

%% Binary prediction (based on th multiclass)
fprintf('Projecting the multiclass prediction to the binary space.\n');
Ytest = (Ytest ~= 4);
fprintf('Saving the resulting binary prediction.\n');
save('pred_binary.mat', 'Ytest');
