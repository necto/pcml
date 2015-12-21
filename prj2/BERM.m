function [ ber ] = BERM( y, ypred )
  % Compute the Balanced Error Rate (BER) and print misprediction for 
  % each class to help understand misclassification and improve model
  C = 4;    % Number of categories
  ber = 0;
  fprintf('\nBER for class:');
  for i = 1:C
    idx = find(y==i);
    Nc = size(idx, 1);  % number of examples in class c
    BERClassI = (sum(y(idx) ~= ypred(idx)) / Nc);
    ber = ber + BERClassI;
        
    % Print BER for each class, as well as the misclassification
    fprintf('\n%d: %.2f%% (Nc:%d, A:%d, C:%d, H:%d, O:%d)', i, ...
    100*BERClassI, Nc, sum( ypred(idx) == 1), sum( ypred(idx) == 2),...
    sum( ypred(idx) == 3),sum( ypred(idx) == 4));
  end;
  ber = ber/C;
end

