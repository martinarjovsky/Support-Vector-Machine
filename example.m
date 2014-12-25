clearvars;

m = 1000;
n = 2;

X = -2 + 4*rand([m, n]);
Y = 2*(sum(abs(X), 2) <= 1) - 1;

sigma = 1;
C = 1;

[ alpha, b ] = SVMtrain_QP( X, Y, C, sigma);

hold all
plot(X(Y>=0,1), X(Y>=0,2), '.');
plot(X(Y<0,1), X(Y<0,2), '.');
title('True targets');

Ypred = zeros(m, 1);
for i = 1:m
    Ypred(i) = SVMpredict(X(i, :), alpha, b, X, Y, sigma);
end

figure;
hold all
plot(X(Ypred>=0,1), X(Ypred>=0,2), '.');
plot(X(Ypred<0,1), X(Ypred<0,2), '.');
title('Predicted targets');