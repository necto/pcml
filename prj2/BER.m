function [ ber ] = BER( y, ypred )
  % Compute the Balanced Error Rate (BER)
  C = 4; % Number of classes
  ber = 0;
  for i = 1:C
    idx = find(y==i);
    Nc = size(idx, 1);
    ber = ber + (sum(y(idx) ~= ypred(idx)) / Nc);
  end;
  ber = ber/C;
end

