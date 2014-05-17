function [margin] = SVMpredict( x, alpha, b, X, Y)
% Given the optimal dual variables of the SVM problem, the bias,
%   the training data, the training labels and a data point x, it
%   calculates the margin for x
    m = length(alpha);
    prod = zeros(m, 1);
    for i = 1:m
        prod(i) = RBF(x, X(i,:));
    end
            
    margin = b + alpha' * (Y.*prod);

end

