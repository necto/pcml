function [ beta ] = ridgeReg( y, X )
  degree = 7;
  proportion = 0.9;
  lambda = -1e5:10:1e5;
	
  % get train and test data
	[XTr, yTr, XTe, yTe] = split( y, X, proportion);
  
	% form tX
	tXTr = [ones(length(yTr), 1) myPoly(XTr(:,1), degree)];
	tXTe = [ones(length(yTe), 1) myPoly(XTe(:,1), degree)];

	% ridgeRegression
  for i = 1:length(lambda)
    beta_i = ridgeRegression(yTr, tXTr, lambda(i));
    
    % train and test MSE
    rmseTr = sqrt(2*computeCost(yTr,tXTr,beta_i)); 
    rmseTe = sqrt(2*computeCost(yTe,tXTe,beta_i)); 
    if (i > 1 && rmseTe < prev_rmseTe)
      beta = beta_i;
      fprintf('Train RMSE :%0.4f Test RMSE :%0.4f\n', rmseTr, rmseTe);
    end
    prev_rmseTe = rmseTe;
  end
end

