function [ hog_negs1 hog_negs2 hog_negs3 ] = mineNegativeFeatures(Model1, Model2, Model3, otherClass, train)
    other_class = find(train.y == 4);
    
    slide_step = 7;
    steps_number = 3;
    
    hog_neg_features = zeros(length(other_class), steps_number, steps_number, 5408);
    
    fprintf('processing %d images\n', length(other_class));
    parfor cli=1:length(other_class)
        id = other_class(cli);

        % load img
        img = imread( sprintf('train/imgs/train%05d.jpg', id) );

        for i=1:steps_number
            for j=1:steps_number
                dx = i*slide_step;
                dy = j*slide_step;
                rect = [dx dy (size(img, 1) - slide_step*steps_number) (size(img, 2) - slide_step*steps_number)];
                subimg = imresize(imcrop(img, rect), [size(img, 1) size(img, 2)]);
                feature = hog( single(subimg)/255, 17, 8);
                feature = feature(:)';
                hog_neg_features(cli, i, j, :) = feature;
            end
        end
    end
    fprintf('%d images processed, yielding %d new points \nFiltering\n', length(other_class), steps_number*steps_number*length(other_class));
    
    hog_neg_features = reshape(hog_neg_features, length(other_class)*steps_number*steps_number, 5408);
    
    [norm, ~, ~] = zscore(hog_neg_features);
    preds1 = predict(Model1, norm);
    preds2 = predict(Model2, norm);
    preds3 = predict(Model3, norm);
    hog_negs1 = hog_neg_features(preds1 ~= otherClass ,:);
    hog_negs2 = hog_neg_features(preds2 ~= otherClass ,:);
    hog_negs3 = hog_neg_features(preds3 ~= otherClass ,:);
    fprintf('%d, %d, %d false positives were left as negative points\n',...
            size(hog_negs1, 1), size(hog_negs2, 1), size(hog_negs3, 1));
end