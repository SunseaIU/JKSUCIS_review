function H = centeringMatrix(dim)

H = -repmat(1./dim, dim, dim) + speye(dim);