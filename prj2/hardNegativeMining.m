clearvars;
close all;
% Load features and labels of training data
%load train/small.mat;
%train = small;
load train/train.mat

% Normalize the data
[train.X_hog, mu, sigma] = zscore(train.X_hog);

%% Mine hard-negative data points
ks3 = 100;
bc3 = 2.6367;
bias3 = 7.017;

C1 = (train.y == 1)*2 - 1;
C2 = (train.y == 2)*2 - 1;
C3 = (train.y == 3)*2 - 1;
fprintf('building 3 models for the exsiting data\n');
SVMModel1 = fitcsvm(train.X_hog, C1, 'KernelFunction', 'rbf', 'KernelScale', ks3, 'BoxConstraint', bc3, 'Cost', [0 1; bias3 0]);
SVMModel2 = fitcsvm(train.X_hog, C2, 'KernelFunction', 'rbf', 'KernelScale', ks3, 'BoxConstraint', bc3, 'Cost', [0 1; bias3 0]);
SVMModel3 = fitcsvm(train.X_hog, C3, 'KernelFunction', 'rbf', 'KernelScale', ks3, 'BoxConstraint', bc3, 'Cost', [0 1; bias3 0]);
negs1 = mineNegativeFeatures(SVMModel1, -1);
negs2 = mineNegativeFeatures(SVMModel2, -1);
negs3 = mineNegativeFeatures(SVMModel3, -1);
save('train/negs1.mat', 'negs1');
save('train/negs2.mat', 'negs2');
save('train/negs3.mat', 'negs3');
