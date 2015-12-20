function [ ber ] = BER( y, ypred, C )
  % Compute the Balanced Error Rate (BER)
  % C = Number of classes
  ber = 0;

  % Binary classification
  if C==2
    if(length(unique(y(:))) == 4)
       y(y < 4) = 2;   % contain an object in {Car,Horse,Airplane}
       y(y == 4) = 1;  % other object
    end
    if(length(unique(ypred(:))) == 4)
      ypred(ypred < 4) = 2;
      ypred(ypred == 4) = 1;
    end
  end 
  classes=unique([y(:) ypred(:)]);
  if(length(classes) ~= C)
    classes = 1:C;
  end
  fprintf('\nBER for class:');
  for i = 1:C
    c = classes(i);
    idx = find(y==c);
    Nc = size(idx, 1);
    BERClassC = (sum(y(idx) ~= ypred(idx)) / Nc);
    ber = ber + BERClassC;
    fprintf('\n%d: %.2f%% (%d)\n', c, 100*BERClassC, Nc );
  end;
  ber = ber/C;
end

