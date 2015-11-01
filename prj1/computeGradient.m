% Linear regression gradient for the (tX, y) dataset in the point beta.
function [ g ] = computeGradient( y, tX, beta )
    N = length(y);
    %compute error
    e = y - tX * beta;
    g = (transpose(tX)*e/(-N));
end

