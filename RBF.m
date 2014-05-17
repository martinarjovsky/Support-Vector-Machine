function [ K ] = RBF(x, y, sigma)
% Radial basis function (Gaussian) kernel

    K = exp(-norm(x - y)^2/sigma);
end

