function [ ber ] = BER( y, ypred, C )
  % Compute the Balanced Error Rate (BER)
  % C = Number of classes
  ber = 0;
  
  % Binary classification
  if C==2
    if(length(unique(ypred(:))) == 4)
        y(y < 4) = 2;   % contain an object in {Car,Horse,Airplane}
        y(y == 4) = 1;  % other object
    end
    if(length(unique(ypred(:))) == 4)
      ypred(ypred < 4) = 2;
      ypred(ypred == 4) = 1;
    end
  end 
  fprintf('\nBER for class:');
  for i = 1:C
    idx = find(y==i);
    Nc = size(idx, 1);
    BERClassI = (sum(y(idx) ~= ypred(idx)) / Nc);
    ber = ber + BERClassI;
    fprintf('\n%d: %.2f%%\n', i, 100*BERClassI );
  end;
  ber = ber/C;
end

