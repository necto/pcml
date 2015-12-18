function [ ber ] = BER( y, ypred, C )
  % Compute the Balanced Error Rate (BER)
  % C = Number of classes
  ber = 0;
  
  % Binary classification
  if C==2
    y(y < 4) = 2;   % contain an object in {Car,Horse,Airplane}
    y(y == 4) = 1;  % other object
  end 
  for i = 1:C
    idx = find(y==i);
    Nc = size(idx, 1);
    ber = ber + (sum(y(idx) ~= ypred(idx)) / Nc);
  end;
  ber = ber/C;
end

