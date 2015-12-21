load test/test.mat;

fprintf('Performing multiclass prediction with the ensemble predictor.\n')
[Ytest, conf] = EnsemblePredict(test);

distr = histc(Ytest, [1 2 3 4]);
fprintf('Classifier found: %d Airplanes(1), %d Cars(2), %d Horses(3), %d Other(4).\n', ...
        distr(1), distr(2), distr(3), distr(4));

fprintf('Saving the predicted vector.\n');
save('pred_multiclass.mat', 'Ytest');

fprintf('Projecting the multiclass prediction to the binary space.\n');
Ytest = (Ytest ~= 4);
fprintf('Saving the resulting binary prediction.\n');
save('pred_binary.mat', 'Ytest');
