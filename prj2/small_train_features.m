addpath(genpath('./piotr_toolbox'));
load train/train.mat;
small = struct('X_hog', zeros(6000,288), 'X_cnn', train.X_cnn, 'y', train.y);

small_HOG = zeros(6000,288);
parfor i=1:6000

    % load img
    img = imread( sprintf('train/small_imgs/train%05d.jpg', i) );

    feature = hog( single(img)/255, 17, 8); % reduce binsize(17)?
    feature = feature(:)';
    
    small_HOG(i, :) = feature;
end

small.X_hog = small_HOG;

save('train/small.mat', 'small');