clearvars;
close all;
% Load features and labels of training data
%load train/small.mat;
%train = small;
load train/train.mat
% Load features of testing data
% load test.mat;
%addpath(genpath('./piotr_toolbox'));

%% Prepare the data
% split randomly into train/test, use K-fold
fprintf('Splitting into train/test..\n');
K = 3;
N = size(train.y, 1);
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K, Nk);
for k = 1:K
    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
end;

% Normalize the data
[train.X_hog, mu, sigma] = zscore(train.X_hog);

%% HOG SVM prediction
if (true)
        tic
        TeBERSub = zeros(K, 1);
        TrBERSub = zeros(K, 1);
        ks = 100;
        bc = 2.6367;
        bias = 7.017;
        parfor k = 1:K
            [Tr, Te] = split4crossValidation(k, idxCV, train);
            Tr_horses = (Tr.y == 3);
            Tr_horses = Tr_horses*2 - 1;
            Te_horses = (Te.y == 3);
            Te_horses = Te_horses*2 - 1;
            SVMModel = fitcsvm(Tr.X_hog, Tr_horses, 'KernelFunction', 'rbf', 'KernelScale', ks, 'BoxConstraint', bc, 'Cost', [0 1; bias 0] );
            [predTe, scoreTe] = predict(SVMModel, Te.X_hog);
            [predTr, scoreTr] = predict(SVMModel, Tr.X_hog);
            Te_horses(Te_horses == -1) = 2;
            Tr_horses(Tr_horses == -1) = 2;
            predTe(predTe == -1) = 2;
            predTr(predTr == -1) = 2;
            TeBERSub(k) = BER(Te_horses, predTe, 2);
            TrBERSub(k) = BER(Tr_horses, predTr, 2);
        end
        berTe = mean(TeBERSub);
        berTr = mean(TrBERSub);
        toc
        display(berTe);
end

optimizing_biases = false;
if(optimizing_biases)
    rng(1) % platform dependent!!
    biases = logspace(-1, 3, 40);
    bers = zeros(length(biases), 1);
    ks = 100;
    bc = 2.6367;
    for biasi = 1:length(biases)
        bias = biases(biasi);
        TeBERSub = zeros(K, 1);
        TrBERSub = zeros(K, 1);
        predSub = cell(K);
        for k = 1:K
            [Tr, Te] = split4crossValidation(k, idxCV, train);
            Tr_horses = (Tr.y == 3);
            Tr_horses = Tr_horses*2 - 1;
            Te_horses = (Te.y == 3);
            Te_horses = Te_horses*2 - 1;
            SVMModel = fitcsvm(Tr.X_hog, Tr_horses, 'KernelFunction', 'rbf', 'KernelScale', ks, 'BoxConstraint', bc, 'Cost', [0 1; bias 0] );
            [predTe, scoreTe] = predict(SVMModel, Te.X_hog);
            [predTr, scoreTr] = predict(SVMModel, Tr.X_hog);
            Te_horses(Te_horses == -1) = 2;
            Tr_horses(Tr_horses == -1) = 2;
            predTe(predTe == -1) = 2;
            predTr(predTr == -1) = 2;
            TeBERSub(k) = BER(Te_horses, predTe, 2);
            TrBERSub(k) = BER(Tr_horses, predTr, 2);
          end
          berTe(biasi) = mean(TeBERSub);
          berTr(biasi) = mean(TrBERSub);
          fprintf('bias: %d train BER: %d, test BER: %d\n', bias, berTr(biasi), berTe(biasi));
    end
    semilogx(biases,berTe);
    hold on;
    semilogx(biases,berTr);
end

optimize_box_constraints = false;
if (optimize_box_constraints)
    rng(1) % platform dependent!!
    box_constraints = logspace(-1, 1, 20);
    bers = zeros(length(box_constraints), 1);
    ks = 100;
    bias = 7.0170;
    for bci = 1:length(box_constraints)
        bc = box_constraints(bci);
        TeBERSub = zeros(K, 1);
        TrBERSub = zeros(K, 1);
        predSub = cell(K);
        for k = 1:K
            [Tr, Te] = split4crossValidation(k, idxCV, train);
            Tr_horses = (Tr.y == 3);
            Tr_horses = Tr_horses*2 - 1;
            Te_horses = (Te.y == 3);
            Te_horses = Te_horses*2 - 1;
            SVMModel = fitcsvm(Tr.X_hog, Tr_horses, 'KernelFunction', 'rbf', 'KernelScale', ks, 'BoxConstraint', bc, 'Cost', [0 1; bias 0] );
            [predTe, scoreTe] = predict(SVMModel, Te.X_hog);
            [predTr, scoreTr] = predict(SVMModel, Tr.X_hog);
            Te_horses(Te_horses == -1) = 2;
            Tr_horses(Tr_horses == -1) = 2;
            predTe(predTe == -1) = 2;
            predTr(predTr == -1) = 2;
            TeBERSub(k) = BER(Te_horses, predTe, 2);
            TrBERSub(k) = BER(Tr_horses, predTr, 2);
        end
          berTe(bci) = mean(TeBERSub);
          berTr(bci) = mean(TrBERSub);
          fprintf('box constraint: %d train BER: %d, test BER: %d\n', bc, berTr(bci), berTe(bci));
    end
    semilogx(box_constraints,berTe);
    hold on;
    semilogx(box_constraints,berTr);
end

plot_kernel_scale_dependency = false;
if (plot_kernel_scale_dependency) 
    rng(1) % platform dependent!!
    kernel_scales = logspace(1, 3, 10);
    for ksi = 1:length(kernel_scales)
        ks = kernel_scales(ksi);
        TeBERSub = zeros(K, 1);
        TrBERSub = zeros(K, 1);
        %ks = 100;
        bc = 2.6367;
        bias = 7.017;
        for k = 1:K
            [Tr, Te] = split4crossValidation(k, idxCV, train);
            Tr_horses = (Tr.y == 3);
            Tr_horses = Tr_horses*2 - 1;
            Te_horses = (Te.y == 3);
            Te_horses = Te_horses*2 - 1;
            SVMModel = fitcsvm(Tr.X_hog, Tr_horses, 'KernelFunction', 'rbf', 'KernelScale', ks, 'BoxConstraint', bc, 'Cost', [0 1; bias 0] );
            [predTe, scoreTe] = predict(SVMModel, Te.X_hog);
            [predTr, scoreTr] = predict(SVMModel, Tr.X_hog);
            Te_horses(Te_horses == -1) = 2;
            Tr_horses(Tr_horses == -1) = 2;
            predTe(predTe == -1) = 2;
            predTr(predTr == -1) = 2;
            TeBERSub(k) = BER(Te_horses, predTe, 2);
            TrBERSub(k) = BER(Tr_horses, predTr, 2);
          end
          berTe(ksi) = mean(TeBERSub);
          berTr(ksi) = mean(TrBERSub);
          fprintf('kernel scale: %d train BER: %d, test BER: %d\n', ks, berTr(ksi), berTe(ksi));
    end
    semilogx(kernel_scales,berTe);
    hold on;
    semilogx(kernel_scales,berTr);
end