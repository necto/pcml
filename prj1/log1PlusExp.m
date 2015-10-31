% A numerically optimized formula log(1 + exp(x)), approximated by exp(x)
% in the lower end, and by an identity x in ther upper end. Inspired by
% the fact, that in MatLab exp(710) = Inf, so the function is screwed
% already in the order of a thousand.
function [ y ] = log1PlusExp( x )
    y(x >= 20) = x(x >= 20);
    y(x <= -10) = exp(x(x <= -10));
    y(-10 < x & x < 20) = log(1 + exp(x(-10 < x & x < 20)));
    y = y';
end

