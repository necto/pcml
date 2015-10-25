% PCML - lab1 18.09.2015
clear all;
close all;

load('data/regression.mat');
load('data/classification.mat');

% % normalize features (store the mean and variance)
% x = height;
% meanX = mean(x);
% x = x - meanX;
% stdX = std(x);
% x = x./stdX;

%% Histograms
% f = figure;
% histogram(y_train,2);
% hx = xlabel('y\_train');
% 
% f2 = figure;
% for i = 1:24
%     subplot(6,4,i);
%     histogram(X_train(:,i));
%     lbl = sprintf('X\\_train %d',i);
%     hx = xlabel(lbl);
% end
% f3 = figure;
% for i = 1:24
%     subplot(6,4,i);
%     histogram(X_test(:,i));
%     lbl = sprintf('X\\_test %d',i);
%     hx = xlabel(lbl);
% end

  
% Form (y,tX) to get regression data in matrix form
tX_train = horzcat(ones(length(X_train),1), X_train);
alpha = 0.000001; % Step size
beta = leastSquaresGD(y_train, tX_train, alpha);
lambda = 1e-5;
%beta = ridgeRegression(y_train, tX_train, lambda);

alpha = 0.000001;
lambda = 1e-5;
beta = penLogisticRegression(y_train, tX_train, alpha, lambda)
