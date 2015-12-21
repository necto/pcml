function showWrongPredictions( Te, ypred )
% Display images for which we predict the wrong label.
  s = 5;  % There are s*s images per figure
  n = 1;
  figure;
  for i = 1:size(ypred,1)
    % Show only mispredicted images
    if (Te.y(i) ~= ypred(i))
      % Group s*s images per figure
      if(n == s*s+1)
        figure;
        n = 1;
      end
      subplot(s, s, n)
      img = imread( sprintf('train/imgs/train%05d.jpg', Te.idxs(i)) );
      imshow(img);
      n = n + 1;
    end
  end
end

