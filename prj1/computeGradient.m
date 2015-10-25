function [ g ] = computeGradient( y, tX, beta )
    N = length(y);
    %compute error
    e = y - tX * beta;
    g = (transpose(tX)*e/(-N));
end

