function [w,b] = qplearnsvm(X,Y,C)

[a,b] = solvesvm(X*X',Y,C);
w = X'*a;
