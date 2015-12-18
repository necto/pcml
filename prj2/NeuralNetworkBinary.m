function [ ber ] = NeuralNetworkBinary( Tr, Te )
addpath(genpath('./DeepLearnToolbox'))

rng(8339);  % fix seed, this    NN may be very sensitive to initialization

% setup NN. The first layer needs to have number of features neurons,
%  and the last layer the number of classes (here four).
nn = nnsetup([size(Tr.X_cnn,2) 10 2]);
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples

% if == 1 => plots trainin error as the NN is trained
opts.plot               = 0;

nn.learningRate = 2;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor( size(Tr.X_cnn) / opts.batchsize);
Tr.X_cnn = Tr.X_cnn(1:numSampToUse,:);
Tr.y = Tr.y(1:numSampToUse);

% normalize data
[Tr.normX, mu, sigma] = zscore(Tr.X_cnn); % train, get mu and std

% prepare labels for NN
LL = [1*(Tr.y == 4) , ...
      1*(Tr.y < 4) ];  % first column, p(y=1)
                        % second column, p(y=2), etc

[nn, L] = nntrain(nn, Tr.normX, LL, opts);


Te.normX = normalize(Te.X_cnn, mu, sigma);  % normalize test data

% to get the scores we need to do nnff (feed-forward)
%  see for example nnpredict().
% (This is a weird thing of this toolbox)
nn.testing = 1;
nn = nnff(nn, Te.normX, zeros(size(Te.normX,1), nn.size(end)));
nn.testing = 0;


% predict on the test set
nnPred = nn.a{end};

% get the most likely class
[~,classVote] = max(nnPred,[],2);

ber = BER(Te.y, classVote, 2);
fprintf('\nBER: %.2f%\n\n', ber );
end

