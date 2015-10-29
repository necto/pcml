function [ y ] = log1PlusExp( x )
    y(x >= 20) = x(x >= 20);
    y(x <= -10) = exp(x(x <= -10));
    y(-10 < x & x < 20) = log(1 + exp(x(-10 < x & x < 20)));
    y = y';
end

