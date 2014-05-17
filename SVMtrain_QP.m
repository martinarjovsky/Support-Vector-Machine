function [ alpha, b ] = SVMtrain_QP( X, Y, C )
% SVM-QP Solves de dual quadratic program of the l1-regularized SVM problem
% It solves the QP using CVX. The return variables are the dual variables
% of the SVM problem, which are then used for testing.

    m = size(X,2);
    K = zeros(m);
    for i = 1:m
        for j = 1:m
            K(i,j) = RBF(X(i,:), X(j,:)) * Y(i) * Y(j);
        end
    end
    one = ones(m,1);
    alpha = zeros(m,1);
    cvx_begin
        variables alpha(m)
        maximize(-1/2 * alpha'*K*alpha + one'*alpha)
        subject to
            0 <= alpha <= C
            alpha'*Y == 0
    cvx_end

    
    
end

