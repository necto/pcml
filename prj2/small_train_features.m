addpath(genpath('./piotr_toolbox'));
load train/train.mat;
small = struct('X_hog', zeros(6000,648), 'X_cnn', train.X_cnn, 'y', train.y);

for i=1:6000

    % load img
    img = imread( sprintf('train/small_imgs/train%05d.jpg', i) );

    feature = hog( single(img)/255, 17, 18); % reduce binsize(17)?
    feature = feature(:)';
    
    small.X_hog(i, :) = feature;
end

save('train/small.mat', 'small');