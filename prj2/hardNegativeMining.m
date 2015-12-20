clearvars;
close all;
% Load features and labels of training data
%load train/small.mat;
%train = small;
load train/train.mat
addpath(genpath('./piotr_toolbox'));

%% Mine hard-negative data points


svmC1 = load('models/svmC1.mat', 'SVMModel');
svmC1 = svmC1.('SVMModel');
svmC2 = load('models/svmC2.mat', 'SVMModel');
svmC2 = svmC2.('SVMModel');
svmC3 = load('models/svmC3.mat', 'SVMModel');
svmC3 = svmC3.('SVMModel');
[negs1, negs2, negs3] = mineNegativeFeatures(svmC1, svmC2, svmC3, -1, train);
save('train/negs1.mat', 'negs1');
save('train/negs2.mat', 'negs2');
save('train/negs3.mat', 'negs3');
