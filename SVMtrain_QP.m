
function [ alpha, b ] = SVMtrain_QP( X, Y, C, sigma)
% SVM-QP Solves de dual quadratic program of the L1-regularized SVM problem
% It solves the QP using CVX. The return variables are the dual variables
% of the SVM problem, which are then used for testing.

    m = size(X,1);
    K = zeros(m);
    for i = 1:m
        for j = 1:m
            K(i,j) = RBF(X(i,:), X(j,:), sigma);
        end
    end
    one = ones(m,1);
    alpha = zeros(m,1);
    cvx_begin
        variables alpha(m)
        maximize(-1/2 * (alpha.*Y)'*K*(alpha.*Y) + one'*alpha)
        subject to
            0 <= alpha <= C
            alpha'*Y == 0
    cvx_end

    % To calculate b we pick an arbitrary support vector
    ind = find(alpha>=C*0.00001 & alpha<=C*(1-0.00001));
    ind = ind(1);
    
    b = Y(ind) - alpha'*(Y.*K(:,ind));   
    
    
end

