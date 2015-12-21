if (exist('reportMode', 'var') == 0)
    clearvars;
    close all;
    % Possible values: 
    % measure_BER             : use CV to measure the SVM performance
    % produce_model           : use the full training set to train and
    %                           save an SVM model
    % optimize_bias           : search for an optimal bias parameter
    %                           (manual)
    % optimize_box_constraint : search for an optimal box constraint
    %                          (manual)
    % optimize_kernel_scale   : search for an optimal kernel scale (manual)
    %                           also if printFigure is true, saves the
    %                           figure useful for the report.
    task = 'measure_BER';
    
    % The class we are working with.
    % 1 - Aeroplane; 2 - Car; 3 - Horse
    positiveClass = 1;
    
    % Use the hard-negative mined additional training set
    useNegs = false;
    
    % Print the generated figure for the report. 
    printFigure = true;
    
    % Use a small features instead. For performance only.
    useSmalls = false;
    
    % The Cross Validation number of folds.
    K = 3;
else
    % Some default values for the report mode.
    if (exist('positiveClass', 'var') == 0)
        positiveClass = 1;
    end
    if (exist('useNegs', 'var') == 0)
        useNegs = true;
    end
    if (exist('printFigure', 'var') == 0)
        printFigure = true;
    end
    if (exist('useSmalls', 'var') == 0)
        useSmalls = false;
    end
    if (exist('K', 'var') == 0)
        K = 3;
    end
    
    % You have to specify the 'task' explicitly. Look at the section above
    % for the possible values.
end;


% Load features and labels of training data
if (useSmalls)
    load train/small.mat;
    train = small;
else
    load train/train.mat;
end

% Load the hard-negative additional training data.
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

% Get the optimal SVM parameters.
[optimalKernelScale, optimalBoxConstraint, optimalBias] = ...
    OptimalSVMParams(positiveClass, useNegs);

% Decipher what we are going to do this time.
measure_BER = strcmp(task, 'measure_BER');
produce_model = strcmp(task, 'produce_model');
optimize_bias = strcmp(task, 'optimize_bias');
optimize_box_constraint = strcmp(task, 'optimize_box_constraint');
optimize_kernel_scale = strcmp(task, 'optimize_kernel_scale');

if (~(measure_BER || produce_model || optimize_bias || ...
      optimize_box_constraint || optimize_kernel_scale))
  fprintf('%s is not a valid task\n', task);
end


%% Prepare the data
% Split randomly into train/test, use K-fold
fprintf('Splitting into train/test..\n');
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
if (measure_BER)
    fprintf('Measuring performance for class %d\n', positiveClass);
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
    fprintf('Test BER: %d (std %d), Train BER: %d (std %d)\n', ...
            berTe, stdTe, berTr, stdTr);
end

%% Produce and save the model
if (produce_model)
    ks = optimalKernelScale;
    bc = optimalBoxConstraint;
    bias = optimalBias;
    fprintf('Producing model for class %d\n', positiveClass);
    train_labels = (train.y == positiveClass)*2 - 1;
    fprintf('Creating a model on the full train dataset\n');
    SVMModel = fitcsvm([train.X_hog; negs], [train_labels; negLabels], ...
                       'KernelFunction', 'rbf', 'KernelScale', ks, ...
                       'BoxConstraint', bc, 'Cost', [0 1; bias 0] );
    fprintf('Saving the model\n');
    modelFileName = sprintf('models/svmC%d.mat', positiveClass);
    save(modelFileName, 'SVMModel');
end

%% Optimize the bias parameter
if(optimize_bias)
    fprintf('Optimizing bias\n');
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

%% Optimize the Box Constraint of SVM
if (optimize_box_constraint)
    fprintf('Optimizing box constraint\n');
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

%% Optimize the kernel scale
if (optimize_kernel_scale)
    fprintf('Optimizing kernel scale\n');
    rng(1) % platform dependent!!
    kernel_scales = logspace(1, 3, 40);
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
        
    hx = xlabel('Kernel scale');
    hy = ylabel('BER');
    legend('Test error', 'Training error', 'Location', 'northwest');
    set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
    set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
    grid on;

    if (printFigure)
        disp('printing the figure');
        set(gcf, 'PaperUnits', 'centimeters');
        set(gcf, 'PaperPosition', [0 0 20 12]);
        set(gcf, 'PaperSize', [20 12]);
        print -dpdf 'report/figures/kernel_scale_ber.pdf'
    end;
end
