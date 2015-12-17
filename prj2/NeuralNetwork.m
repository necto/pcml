function [ ber ] = NeuralNetwork( Tr, Te )

fprintf('Training simple neural network..\n');
addpath(genpath('./DeepLearnToolbox'))

rng(8339);  % fix seed, this    NN may be very sensitive to initialization

% setup NN. The first layer needs to have number of features neurons,
%  and the last layer the number of classes (here four).
nn = nnsetup([size(Tr.X_hog,2) 10 4]);
opts.numepochs =  20;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples

% if == 1 => plots trainin error as the NN is trained
opts.plot               = 0;

nn.learningRate = 2;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor( size(Tr.X_hog) / opts.batchsize);
Tr.X_hog = Tr.X_hog(1:numSampToUse,:);
Tr.y = Tr.y(1:numSampToUse);

% normalize data
[Tr.normX, mu, sigma] = zscore(Tr.X_hog); % train, get mu and std

% prepare labels for NN
LL = [1*(Tr.y == 1), ...
      1*(Tr.y == 2), ...
      1*(Tr.y == 3), ...
      1*(Tr.y == 4) ];  % first column, p(y=1)
                        % second column, p(y=2), etc

[nn, L] = nntrain(nn, Tr.normX, LL, opts);


Te.normX = normalize(Te.X_hog, mu, sigma);  % normalize test data

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

% get overall error [NOTE!! this is not the BER, you have to write the code
%                    to compute the BER!]
predErr = sum( classVote ~= Te.y ) / length(Te.y);
fprintf('\nTesting error: %.2f%%\n', predErr * 100 );

ber = BER(Te.y, classVote);
fprintf('\nBER: %.2f%\n\n', ber );
end
