function [ ber ] = BERM( y, ypred )
  % Compute the Balanced Error Rate (BER)
  C = 4;
  ber = 0;
  fprintf('\nBER for class:');
  for i = 1:C
    idx = find(y==i);
    Nc = size(idx, 1);
    BERClassI = (sum(y(idx) ~= ypred(idx)) / Nc);
    ber = ber + BERClassI;
    
    err = zeros(C, 1);
    
    idxpred = find(ypred==i);
    Np = size(idxpred, 1);
    fprintf('\n%d: %.2f%% (Nc:%d, Np:%d, A:%d, C:%d, H:%d, O:%d)', i, ...
    100*BERClassI, Nc, Np, sum( y(idxpred) == 1), sum( y(idxpred) == 2),...
    sum( y(idxpred) == 3),sum( y(idxpred) == 4));
  end;

  ber = ber/C;
end

