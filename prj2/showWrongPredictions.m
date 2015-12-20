function showWrongPredictions( Te, ypred )
  sum(Te.y ~= ypred)
  s = 5;
  n = 1;
  figure;
  for i = 1:size(ypred,1)
    if (Te.y(i) ~= ypred(i))
      if(n == s*s+1)
        figure;
        n = 1;
      end
      subplot(s, s, n)
      img = imread( sprintf('train/imgs/train%05d.jpg', Te.idxs(i)) );
      imshow(img);
      %title(sprintf('Label: %d, Pred: %d', Te.y(i), ypred(i)));
      n = n + 1;
    end
  end
end

