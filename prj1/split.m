% Shuffle and split the given data (X, y) into two groups: test (XTe, yTe)
% and training (XTr, yTr) with the given ratio:
% prop = length(yTe)/(length(yTe) + length(yTr)).
% The last parameter if the rundom number generator seed for shuffling the
% data.
function [XTr, yTr, XTe, yTe] = split(y, X, prop, seed)
% split the data into train and test given a proportion
    setSeed(seed);
    N = size(y,1);
    % generate random indices
    idx = randperm(N);
    Ntr = floor(prop * N);
    % select few as training and others as testing
    idxTr = idx(1:Ntr);
    idxTe = idx(Ntr+1:end);
    % create train-test split
    XTr = X(idxTr,:);
    yTr = y(idxTr);
    XTe = X(idxTe,:);
    yTe = y(idxTe);
end

function setSeed(seed)
% set the random number generator seed. Use it for reproducibility of pseudo
% random numbers.
	global RNDN_STATE  RND_STATE
	RNDN_STATE = randn('state');
	randn('state',seed);
	RND_STATE = rand('state');
	%rand('state',seed);
	rand('twister',seed);
end
