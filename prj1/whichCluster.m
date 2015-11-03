% Linear regression loss function.
function [ idxTe ] = whichCluster( C, tXTe )
  idxTe = zeros(size(tXTe,1),1);
  dists = zeros(size(tXTe,1),3);
  tX = horzcat(tXTe(:,58),tXTe(:,43));
  dists(:,1) = pdist2(C(1,:),tX);
  dists(:,2) = pdist2(C(2,:),tX);
  dists(:,3) = pdist2(C(3,:),tX);
  [~, idxTe] = min(dists,[],2);
end

