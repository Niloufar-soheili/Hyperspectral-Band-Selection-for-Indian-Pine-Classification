function [img] = Convert3d(img, h, w, numBands)

if (ndims(img) ~= 2)
    error('Input image must be p x N.');
end

[numBands, N] = size(img);

img = reshape(img.', h, w, numBands); 

return;