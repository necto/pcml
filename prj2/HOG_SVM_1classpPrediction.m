clearvars;
close all;
% Load features and labels of training data
%load train/small.mat;
%train = small;
load train/train.mat;


positiveClass = 2;
useNegs = false;

negs = [];
negLabels = [];
if (useNegs)
    if (positiveClass == 1)
        load train/negs1.mat;
        negs = negs1;
    elseif (positiveClass == 2)
        load train/negs2.mat;
        negs = negs2;
    elseif (positiveClass == 3)
        load train/negs3.mat;
        negs = negs3;
    else
        fprintf('wrong positive class: %d\n', positiveClass);
    end
    negLabels = zeros(size(negs, 1), 1) - 1;
    [negs, ~, ~] = zscore(negs);
end

[optimalKernelScale, optimalBoxConstraint, optimalBias] = ...
    OptimalSVMParams(positiveClass, useNegs);


%% Prepare the data
% split randomly into train/test, use K-fold
fprintf('Splitting into train/test..\n');
K = 2;
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
produce_model = true;
if (produce_model)
    fprintf('producing model for class %d\n', positiveClass);
    tic
    TeBERSub = zeros(K, 1);
    TrBERSub = zeros(K, 1);
    ks = optimalKernelScale;
    bc = optimalBoxConstraint;
    bias = optimalBias;
    parfor k = 1:K
        [Tr, Te] = split4crossValidation(k, idxCV, train);
        Tr_labels = (Tr.y == positiveClass)*2 - 1;
        Te_labels = (Te.y == positiveClass)*2 - 1;
        SVMModel = fitcsvm([Tr.X_hog; negs], [Tr_labels; negLabels], ...
                           'KernelFunction', 'rbf', 'KernelScale', ks, ...
                           'BoxConstraint', bc, 'Cost', [0 1; bias 0] );
        [predTe, scoreTe] = predict(SVMModel, Te.X_hog);
        [predTr, scoreTr] = predict(SVMModel, Tr.X_hog);
        TeBERSub(k) = BER(Te_labels, predTe, 2);
        TrBERSub(k) = BER(Tr_labels, predTr, 2);
    end
    berTe = mean(TeBERSub);
    berTr = mean(TrBERSub);
    stdTe = std(TeBERSub);
    stdTr = std(TrBERSub);
    toc
    
    train_labels = (train.y == positiveClass)*2 - 1;
    fprintf('Creating a model on the full train dataset\n');
    SVMModel = fitcsvm([train.X_hog; negs], [train_labels; negLabels], ...
                       'KernelFunction', 'rbf', 'KernelScale', ks, ...
                       'BoxConstraint', bc, 'Cost', [0 1; bias 0] );
    modelFileName = sprintf('models/svmC%d.mat', positiveClass);
    save(modelFileName, 'SVMModel');
    fprintf('test BER: %d (std %d), train BER: %d (std %d)', ...
            berTe, stdTe, berTr, stdTr);
end

optimizing_biases = false;
if(optimizing_biases)
    fprintf('optimizing bias\n');
    rng(1) % platform dependent!!
    biases = logspace(1, 1.5, 6);
    bers = zeros(length(biases), 1);
    ks = optimalKernelScale;
    bc = optimalBoxConstraint;
    for biasi = 1:length(biases)
        bias = biases(biasi);
        TeBERSub = zeros(K, 1);
        TrBERSub = zeros(K, 1);
        predSub = cell(K);
        for k = 1:K
            [Tr, Te] = split4crossValidation(k, idxCV, train);
            Tr_labels = (Tr.y == positiveClass)*2 - 1;
            Te_labels = (Te.y == positiveClass)*2 - 1;
            SVMModel = fitcsvm([Tr.X_hog; negs], [Tr_labels; negLabels], 'KernelFunction', 'rbf', 'KernelScale', ks, 'BoxConstraint', bc, 'Cost', [0 1; bias 0] );
            [predTe, scoreTe] = predict(SVMModel, Te.X_hog);
            [predTr, scoreTr] = predict(SVMModel, Tr.X_hog);
            TeBERSub(k) = BER(Te_labels, predTe, 2);
            TrBERSub(k) = BER(Tr_labels, predTr, 2);
        end
        berTe(biasi) = mean(TeBERSub);
        berTr(biasi) = mean(TrBERSub);
        fprintf('bias: %d train BER: %d, test BER: %d\n', bias, berTr(biasi), berTe(biasi));
    end
    semilogx(biases,berTe);
    hold on;
    semilogx(biases,berTr);
end

optimize_box_constraint = false;
if (optimize_box_constraint)
    fprintf('optimizeing box constraint\n');
    rng(1) % platform dependent!!
    box_constraints = logspace(-0.5, 1, 20);
    bers = zeros(length(box_constraints), 1);
    ks = optimalKernelScale;
    bias = optimalBias;
    for bci = 1:length(box_constraints)
        bc = box_constraints(bci);
        TeBERSub = zeros(K, 1);
        TrBERSub = zeros(K, 1);
        predSub = cell(K);
        for k = 1:K
            [Tr, Te] = split4crossValidation(k, idxCV, train);
            Tr_labels = (Tr.y == positiveClass)*2 - 1;
            Te_labels = (Te.y == positiveClass)*2 - 1;
            SVMModel = fitcsvm([Tr.X_hog; negs], [Tr_labels; negLabels], 'KernelFunction', 'rbf', 'KernelScale', ks, 'BoxConstraint', bc, 'Cost', [0 1; bias 0] );
            [predTe, scoreTe] = predict(SVMModel, Te.X_hog);
            [predTr, scoreTr] = predict(SVMModel, Tr.X_hog);
            TeBERSub(k) = BER(Te_labels, predTe, 2);
            TrBERSub(k) = BER(Tr_labels, predTr, 2);
        end
        berTe(bci) = mean(TeBERSub);
        berTr(bci) = mean(TrBERSub);
        fprintf('box constraint: %d train BER: %d, test BER: %d\n', bc, berTr(bci), berTe(bci));
    end
    semilogx(box_constraints,berTe);
    hold on;
    semilogx(box_constraints,berTr);
end

optimize_kernel_scale = false;
if (optimize_kernel_scale)
    fprintf('optimizing kernel scale\n');
    rng(1) % platform dependent!!
    kernel_scales = logspace(1.5, 2.5, 5);
    for ksi = 1:length(kernel_scales)
        ks = kernel_scales(ksi);
        TeBERSub = zeros(K, 1);
        TrBERSub = zeros(K, 1);
        bc = optimalBoxConstraint;
        bias = optimalBias;
        for k = 1:K
            [Tr, Te] = split4crossValidation(k, idxCV, train);
            Tr_labels = (Tr.y == positiveClass)*2 - 1;
            Te_labels = (Te.y == positiveClass)*2 - 1;
            SVMModel = fitcsvm([Tr.X_hog; negs], [Tr_labels; negLabels], 'KernelFunction', 'rbf', 'KernelScale', ks, 'BoxConstraint', bc, 'Cost', [0 1; bias 0] );
            [predTe, scoreTe] = predict(SVMModel, Te.X_hog);
            [predTr, scoreTr] = predict(SVMModel, Tr.X_hog);
            TeBERSub(k) = BER(Te_labels, predTe, 2);
            TrBERSub(k) = BER(Tr_labels, predTr, 2);
        end
        berTe(ksi) = mean(TeBERSub);
        berTr(ksi) = mean(TrBERSub);
        fprintf('kernel scale: %d train BER: %d, test BER: %d\n', ks, berTr(ksi), berTe(ksi));
    end
    semilogx(kernel_scales,berTe);
    hold on;
    semilogx(kernel_scales,berTr);
end