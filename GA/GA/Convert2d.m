function [M] = Convert2d(M)

if(ndims(M)==3)
    
    if (ndims(M) ~= 3)
        error('Input image must be m x n x p.');
    end
    
    [h, w, numBands] = size(M);
    
    % M = reshape(M, w*h, numBands).';
    M = double(reshape(M, w*h, numBands).');
else
    [h, w] = size(M);
    
    M = double(reshape(M, w*h, 1).');
end
return;