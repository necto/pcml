function outputNNModel( train )
addpath(genpath('./DeepLearnToolbox'))

rng(8339);  % fix seed, this    NN may be very sensitive to initialization

% setup NN. The first layer needs to have number of features neurons,
%  and the last layer the number of classes (here four).
nn = nnsetup([size(train.X_cnn,2) 10 4]);
opts.numepochs =  90;   %  Number of full sweeps through data
opts.batchsize = 200;  %  Take a mean gradient step over this many samples

% if == 1 => plots trainin error as the NN is trained
opts.plot               = 0;

nn.learningRate = 3;

% this neural network implementation requires number of samples to be a
% multiple of batchsize, so we remove some for this to be true.
numSampToUse = opts.batchsize * floor( size(train.X_cnn) / opts.batchsize);
train.X_cnn = train.X_cnn(1:numSampToUse,:);
train.y = train.y(1:numSampToUse);

% normalize data
[train.normX, mu, sigma] = zscore(train.X_cnn); % train, get mu and std

% prepare labels for NN
LL = [1*(train.y == 1), ...
      1*(train.y == 2), ...
      1*(train.y == 3), ...
      1*(train.y == 4) ];  % first column, p(y=1)
                        % second column, p(y=2), etc

[nn, L] = nntrain(nn, train.normX, LL, opts);


save('models/NeuralNetwork', 'nn', 'mu', 'sigma'); 

end

