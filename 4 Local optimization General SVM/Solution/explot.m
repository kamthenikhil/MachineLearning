function explot(trainX,trainY)
% This function picks a random weight vector and intercept (?!)
% and plots the dataset and resulting classifier.
% It isn't useful, but demonstrates how to use plotclassifier


C = 0.1;

c=1;
d=2;

[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));

plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));

end

function Y = testQuadraticKernel(newX,trainX,trainY,alpha,b,c,d)
    
    m = trainY.*alpha;
    Y = quadraticKernel(newX,trainX,c,d)*m + b;
end

function Y = quadraticKernel(A,B,c,d)
    Y = (A*B' + c).^d;
end