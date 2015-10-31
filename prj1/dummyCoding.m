function [ X_dummy ] = dummyCoding( X, columns )
  columns = sort(columns, 'descend');
  X_dummy = X;
  for i =1:length(columns)
    dummycoding = dummyvar(X_dummy(:,columns(i))+1);
    X_dummy = horzcat(X_dummy(:,1:columns(i)-1), dummycoding, X_dummy(:,columns(i)+1:end));
  end
end

