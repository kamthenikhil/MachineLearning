function q2()
% Nikhil Kamthe
% 861245635
% 10/20/2016
% CS 229
% PS 4

tic;
data = load('example1.data','-ascii');

trainX = data(:,1:2);
trainY = data(:,3);

% Figure 1
figure(1);
subplot(3,3,1);
C = 0.1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) linearKernel(A,B));
plotclassifier(trainX,trainY,@(X) testLinearKernel(X,trainX,trainY,alpha,b));
title('C=0.1');
ylabel('Linear');

subplot(3,3,2);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) linearKernel(A,B));
plotclassifier(trainX,trainY,@(X) testLinearKernel(X,trainX,trainY,alpha,b));
title('C=1');

subplot(3,3,3);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) linearKernel(A,B));
plotclassifier(trainX,trainY,@(X) testLinearKernel(X,trainX,trainY,alpha,b));
title('C=10');

c = 1;
d = 2;

subplot(3,3,4);
C = 0.1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));
ylabel('Polynomial c=1, d=2');

subplot(3,3,5);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));

subplot(3,3,6);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));

c = 1;
d = 5;

subplot(3,3,7);
C = 0.1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));
ylabel('Polynomial c=1, d=5');

subplot(3,3,8);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
w = trainX'*(trainY.*alpha);
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));

subplot(3,3,9);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
w = trainX'*(trainY.*alpha);
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));

% Figure 2
figure(2);
subplot(3,3,1);
C = 0.1;
sigma = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));
title('C=0.1');
ylabel('Gauassian \sigma = 1');

subplot(3,3,2);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));
title('C=1');

subplot(3,3,3);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));
title('C=10');

sigma = 5;
subplot(3,3,4);
C = 0.1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));
ylabel('Gauassian \sigma = 5');

subplot(3,3,5);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));

subplot(3,3,6);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));

sigma = 10;
subplot(3,3,7);
C = 0.1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));
ylabel('Gauassian \sigma = 10');

subplot(3,3,8);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));

subplot(3,3,9);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));

data = load('example2.data','-ascii');

trainX = data(:,1:2);
trainY = data(:,3);

% Figure 3
figure(3);
subplot(3,3,1);
C = 0.1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) linearKernel(A,B));
plotclassifier(trainX,trainY,@(X) testLinearKernel(X,trainX,trainY,alpha,b));
title('C=0.1');
ylabel('Linear');

subplot(3,3,2);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) linearKernel(A,B));
plotclassifier(trainX,trainY,@(X) testLinearKernel(X,trainX,trainY,alpha,b));
title('C=1');

subplot(3,3,3);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) linearKernel(A,B));
plotclassifier(trainX,trainY,@(X) testLinearKernel(X,trainX,trainY,alpha,b));
title('C=10');

c = 1;
d = 2;

subplot(3,3,4);
C = 0.1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));
ylabel('Polynomial c=1, d=2');

subplot(3,3,5);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));

subplot(3,3,6);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));

c = 1;
d = 5;

subplot(3,3,7);
C = 0.1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));
ylabel('Polynomial c=1, d=5');

subplot(3,3,8);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));

subplot(3,3,9);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) quadraticKernel(A,B,c,d));
plotclassifier(trainX,trainY,@(X) testQuadraticKernel(X,trainX,trainY,alpha,b,c,d));

% Figure 4
figure(4);
subplot(3,3,1);
C = 0.1;
sigma = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));
title('C=0.1');
ylabel('Gauassian \sigma = 1');

subplot(3,3,2);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));
title('C=1');

subplot(3,3,3);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));
title('C=10');

sigma = 5;
subplot(3,3,4);
C = 0.1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));
ylabel('Gauassian \sigma = 5');

subplot(3,3,5);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));

subplot(3,3,6);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));

sigma = 10;
subplot(3,3,7);
C = 0.1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));
ylabel('Gauassian \sigma = 10');

subplot(3,3,8);
C = 1;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));

subplot(3,3,9);
C = 10;
[alpha,b] = learnsvm(trainX,trainY,C,@(A,B) gaussianKernel(A,B, sigma));
plotclassifier(trainX,trainY,@(X) testGaussianKernel(X,trainX,trainY,alpha,b,sigma));

toc;

end

function Y = testLinearKernel(newX,trainX,trainY,alpha,b)
    % This function computes the y values from the alpha and intercept for
    % a linear kernel
    m = trainY.*alpha;
    Y = linearKernel(newX,trainX)*m + b;
end

function Y = linearKernel(A,B)
    % This function computes the linear kernel
    Y = A*B';
end

function Y = testQuadraticKernel(newX,trainX,trainY,alpha,b,c,d)
    % This function computes the y values from the alpha and intercept for
    % a polynomial kernel
    m = trainY.*alpha;
    Y = quadraticKernel(newX,trainX,c,d)*m + b;
end

function Y = quadraticKernel(A,B,c,d)
    % This function computes the polynomial kernel
    Y = (A*B' + c).^d;
end

function Y = testGaussianKernel(newX,trainX,trainY,alpha,b,sigma)
    % This function computes the y values from the alpha and intercept for
    % a gaussian kernel
    m = trainY.*alpha;
    Y = gaussianKernel(newX,trainX,sigma)*m + b;
end

function Y = gaussianKernel(A,B, sigma)
    % This function computes the gaussian kernel
    [m,d] = size(A);
    [n,d] = size(B);
    Y = zeros(m,n);
    for i = 1:m
        for j = 1:n
            diff = A(i,:) - B(j,:);
            Y(i,j) = exp(-(diff*diff')/(2*sigma^2));
        end
    end
end