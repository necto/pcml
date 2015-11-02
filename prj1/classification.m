if (exist('reportMode', 'var') == 1)
    forReport = true;
else
    clear all;
    forReport = false;
    %Possible values: 'logReg', 'penLogReg', 'removal', 'dummy',
    %'polynomial';
    stage = 'logReg';
end;

load('data/classification.mat');


y_train = (y_train + 1)/2; % !!! remember to invert that for predictions

% dummy-encoding? does not help? or feature #9 is not that bad?
%for i = unique(X_train(:,17))'
%    X_train(:,size(X_train, 2)+1) = (X_train(:,17) == i);
%end

%% Multiple-cut logistic regression.
if (strcmp(stage, 'logReg'))
    seeds = 1:15;
    for s = 1:length(seeds)
        seed = seeds(s);
        [XTr, yTr, XTe, yTe] = split(y_train, X_train, 0.9, seed);
        [XTr, XTr_mean, XTr_std] = normalize(XTr);
        XTe = adjust(XTe, XTr_mean, XTr_std);
        tXTr = [ones(size(XTr, 1), 1) XTr];
        tXTe = [ones(size(XTe, 1), 1) XTe];

        lrBeta = logisticRegression(yTr, tXTr, 1, 1e-5);

        lrY = sigmoid(tXTe * lrBeta) > 0.5;
        [lrTestRMSE(s), lrTest01(s), lrTestLog(s)] = classificationLosses(tXTe, lrBeta, yTe);
        [lrTrainRMSE(s), lrTrain01(s), lrTrainLog(s)] = classificationLosses(tXTr, lrBeta, yTr);
    end;

    lrTest01_std = std(lrTest01)
    lrTestRMSE = mean(lrTestRMSE)
    lrTest01 = mean(lrTest01)
    lrTestLog = mean(lrTestLog)
    
    
    lrTrainRMSE = mean(lrTrainRMSE)
    lrTrain01 = mean(lrTrain01)
    lrTrainLog = mean(lrTrainLog)
end;

[XTr, yTr, XTe, yTe] = split(y_train, X_train, 0.9, 43);
[XTr, XTr_mean, XTr_std] = normalize(XTr);
XTe = adjust(XTe, XTr_mean, XTr_std);
tXTr = [ones(size(XTr, 1), 1) XTr];
tXTe = [ones(size(XTe, 1), 1) XTe];

majority = mode(yTr);

majRMSE = sqrt( (yTe - majority)'*(yTe - majority)/length(yTe))
maj01 = sum(yTe ~= (majority > 0.5))/length(yTe)

K = 7;
N = size(yTr, 1);
idx = randperm(N);
Nk = floor(N/K);
idxCV = zeros(K, Nk);
for k = 1:K
    idxCV(k,:) = idx(1 + (k-1)*Nk:k*Nk);
end;

%% Dummy encoding
if (strcmp(stage, 'dummy'))
    discFeatures = [8 9 10 17];
    for i = 1:length(discFeatures)
        dXTr = tXTr;
        dXTe = tXTe;
        for v = unique(XTr(:, discFeatures(i)))'
            dXTr(:,size(dXTr, 2)+1) = (XTr(:,discFeatures(i)) == v);
            dXTe(:,size(dXTe, 2)+1) = (XTe(:,discFeatures(i)) == v);
        end;
        for k = 1:K
            [yTrTe, yTrTr, dXTrTe, dXTrTr] = split4crossValidation(k, idxCV, yTr, dXTr);

            beta = logisticRegression(yTrTr, dXTrTr, 1e-1, 1e-6);

            [rmseTrSub(k), zeroOneTrSub(k), logTrSub(k)] = classificationLosses(dXTrTr, beta, yTrTr);
            [rmseTeSub(k), zeroOneTeSub(k), logTeSub(k)] = classificationLosses(dXTrTe, beta, yTrTe);
        end;
        rmseTe(i) = mean(rmseTeSub); zeroOneTe(i) = mean(zeroOneTeSub); logTe(i) = mean(logTeSub)
        rmseTr(i) = mean(rmseTrSub); zeroOneTr(i) = mean(zeroOneTrSub); logTr(i) = mean(logTrSub);
        trBeta = logisticRegression(yTr, dXTr, 1e-1, 1e-6);
        [rmseTT(i), zeroOneTT(i), logTT(i)] = classificationLosses(dXTe, trBeta, yTe);
    end;
end;

%% Feature removal
if (strcmp(stage, 'removal'))
    for i = 1:size(tXTr,2)
        errorTeSub = zeros(K, 1);
        errorTrSub = zeros(K, 1);
        rXTr = tXTr(:,[1:i-1 i+1:end]);
        for k = 1:K
            [yTrTe, yTrTr, rXTrTe, rXTrTr] = split4crossValidation(k, idxCV, yTr, rXTr);

            beta = logisticRegression(yTrTr, rXTrTr, 1e-1, 1e-6);
            [rmseTrSub(k), zeroOneTrSub(k), logTrSub(k)] = classificationLosses(rXTrTr, beta, yTrTr);
            [rmseTeSub(k), zeroOneTeSub(k), logTeSub(k)] = classificationLosses(rXTrTe, beta, yTrTe);
        end;
        rmseTe(i) = mean(rmseTeSub); zeroOneTe(i) = mean(zeroOneTeSub); logTe(i) = mean(logTeSub)
        rmseTr(i) = mean(rmseTrSub); zeroOneTr(i) = mean(zeroOneTrSub); logTr(i) = mean(logTrSub);
    end
    [rmseStar irmseStar] = min(rmseTe);
    [zeroOneStar i01Star] = min(zeroOneTe);
    [logStar ilogStar] = min(logTe);

    nfrmseBeta = logisticRegression(yTr, tXTr(:, [1:irmseStar-1 irmseStar+1:end]), 1e-1, 1e-5);
    [nfTestRMSE, ~, ~] = classificationLosses(tXTe(:, [1:irmseStar-1 irmseStar+1:end]), nfrmseBeta, yTe)
    nf01Beta = logisticRegression(yTr, tXTr(:, [1:i01Star-1 i01Star+1:end]), 1e-1, 1e-5);
    [~, noFeatureTest01, ~] = classificationLosses(tXTe(:, [1:i01Star-1 i01Star+1:end]), nf01Beta, yTe)
    nflogBeta = logisticRegression(yTr, tXTr(:, [1:ilogStar-1 ilogStar+1:end]), 1e-1, 1e-5);
    [~, ~, nflogTest] = classificationLosses(tXTe(:, [1:ilogStar-1 ilogStar+1:end]), nflogBeta, yTe)
end;

%% Polynomial
if (strcmp(stage, 'polynomial'))
    mvals = [2];
    alphas = [1e-4];
    lvals = logspace(-3, 0, 70);

    errorTeSub = zeros(K,1);
    errorTrSub = zeros(K,1);
    for j = 1:length(mvals)
        m = mvals(j);
        %pXTr = [ones(size(XTr, 1), 1) myPoly(XTr, m)];
        pXTr = myPoly(tXTr, m);
        pXTe = myPoly(tXTe, m);
        for l = 1:length(lvals)
            lambda = lvals(l);
            for k = 1:K
                [yTrTe, yTrTr, pXTrTe, pXTrTr] = split4crossValidation(k, idxCV, yTr, pXTr);

                alpha = alphas(j);
                beta = penLogisticRegression(yTrTr, pXTrTr, alpha/lambda, lambda, 1e-7/lambda);

                [rmseTrSub(k), zeroOneTrSub(k), logTrSub(k)] = classificationLosses(pXTrTr, beta, yTrTr);
                [rmseTeSub(k), zeroOneTeSub(k), logTeSub(k)] = classificationLosses(pXTrTe, beta, yTrTe);
                [rmseTTSub(k), zeroOneTTSub(k), logTTSub(k)] = classificationLosses(pXTe, beta, yTe);
            end;
            rmseTe(j,l) = mean(rmseTeSub); zeroOneTe(j,l) = mean(zeroOneTeSub); logTe(j,l) = mean(logTeSub)
            rmseTr(j,l) = mean(rmseTrSub); zeroOneTr(j,l) = mean(zeroOneTrSub); logTr(j,l) = mean(logTrSub);
            rmseTT(j,l) = mean(rmseTTSub); zeroOneTT(j,l) = mean(zeroOneTTSub); logTT(j,l) = mean(logTTSub);
        end;
    end;
    [rmseStar lrmseStar] = min(rmseTe(1,:));
    [zeroOneStar l01Star] = min(zeroOneTe(1,:));
    [logStar llogStar] = min(logTe(1,:));

    [cvPolyTestrmse, ~, ~] = classificationLosses(pXTe, penLogisticRegression(yTr, pXTr, 1e-1, lvals(lrmseStar), 1e-4), yTe)
    [~, cvPolyTest01, ~] = classificationLosses(pXTe, penLogisticRegression(yTr, pXTr, 1e-1, lvals(l01Star), 1e-4), yTe)
    [~, ~, cvPolyTestlog] = classificationLosses(pXTe, penLogisticRegression(yTr, pXTr, 1e-1, lvals(llogStar), 1e-4), yTe)

    %pXTe = [ones(size(XTe, 1), 1) myPoly(XTe, 2)];
    %pXTe = myPoly(tXTe, m);

    plot(lvals, logTe, 'b');
    hold on;
    plot(lvals, logTr, 'r');
    set(gca,'XScale', 'log');
    %title('Mispredictions for the second degreee polinom.');
    hx = xlabel('Penalizer coefficient lambda');
    hy = ylabel('logLoss');
    legend('Test error', 'Training error', 'Location', 'SouthEast');
    set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
    set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
    grid on;

    if (forReport)
        disp('printing the figure');
        set(gcf, 'PaperUnits', 'centimeters');
        set(gcf, 'PaperPosition', [0 0 20 12]);
        set(gcf, 'PaperSize', [20 12]);
        print -dpdf 'report/figures/polyLogLoss.pdf'
    end;
end;

%% Penalized logistic regression
if (strcmp(stage, 'penLogReg'))
    lvals = logspace(-7, -1, 100);

    for l = 1:length(lvals)
        lambda = lvals(l);

        for k = 1:K
            [yTrTe, yTrTr, tXTrTe, tXTrTr] = split4crossValidation(k, idxCV, yTr, tXTr);

            alpha = 1e-2/sqrt(sqrt(lambda));
            beta = penLogisticRegression(yTrTr, tXTrTr, alpha, lambda, 1e-7);

            [rmseTrSub(k), zeroOneTrSub(k), logTrSub(k)] = classificationLosses(tXTrTr, beta, yTrTr);
            [rmseTeSub(k), zeroOneTeSub(k), logTeSub(k)] = classificationLosses(tXTrTe, beta, yTrTe);
            [rmseTTSub(k), zeroOneTTSub(k), logTTSub(k)] = classificationLosses(tXTe, beta, yTe);
        end;
        rmseTe(l) = mean(rmseTeSub); zeroOneTe(l) = mean(zeroOneTeSub); logTe(l) = mean(logTeSub)
        rmseTr(l) = mean(rmseTrSub); zeroOneTr(l) = mean(zeroOneTrSub); logTr(l) = mean(logTrSub);
        rmseTT(l) = mean(rmseTTSub); zeroOneTT(l) = mean(zeroOneTTSub); logTT(l) = mean(logTTSub);
    end;
    [rmseStar lrmseStar] = min(rmseTe);
    [zeroOneStar l01Star] = min(zeroOneTe);
    [logStar llogStar] = min(logTe);

    [cvPenTestrmse, ~, ~] = classificationLosses(tXTe, penLogisticRegression(yTr, tXTr, 1e-1, lvals(lrmseStar), 1e-10), yTe)
    [~, cvPenTest01, ~] = classificationLosses(tXTe, penLogisticRegression(yTr, tXTr, 1e-1, lvals(l01Star), 1e-10), yTe)
    [~, ~, cvPenTestlog] = classificationLosses(tXTe, penLogisticRegression(yTr, tXTr, 1e-1, lvals(llogStar), 1e-10), yTe)

    plot(lvals, logTe, 'b');
    hold on;
    plot(lvals, logTr, 'r');
    %hold on;
    %plot(lvals, zeroOneTT, 'g');
    set(gca,'XScale', 'log');
    %title('Mispredictions for penalized logistic regression.');
    hx = xlabel('Penalizer coefficient lambda');
    hy = ylabel('logLoss');
    legend('Test error', 'Training error', 'Location', 'northwest');
    set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
    set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
    grid on;

    if (forReport)
        disp('printing the figure');
        set(gcf, 'PaperUnits', 'centimeters');
        set(gcf, 'PaperPosition', [0 0 20 12]);
        set(gcf, 'PaperSize', [20 12]);
        print -dpdf 'report/figures/penLLmisses.pdf'
    end;
end;

%% Predictions:

if (forReport && strcmp(stage, 'logReg'))
    [XTrn, XTrn_mean, XTrn_std] = normalize(X_train);
    Xtst = adjust(X_test, XTrn_mean, XTrn_std);
    tXTrn = [ones(size(XTrn, 1), 1) XTrn];
    tXtst = [ones(size(Xtst, 1), 1) Xtst];

    lrBeta = logisticRegression(y_train, tXTrn, 0.1, 1e-6);
    predictions = sigmoid(tXtst*lrBeta) > 0.5;
    predictions = predictions*2 - 1;
    csvwrite('predictions_classification.csv', predictions);
    errFile = fopen('test_errors_classification.csv', 'wt');
    fprintf(errFile, '01loss,%d', lrTest01);
    fclose(errFile);
end;
