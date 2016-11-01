function parta()
% Nikhil Kamthe
% 861245635
% 09/28/2016
% CS 229
% PS 1
tic;
plotAverageMeanDistance(2, 'r');
hold on;
plotAverageMeanDistance(5, 'g');
plotAverageMeanDistance(10, 'b');
xlabel('number of data points');
ylabel('average mean distance from nearest neighbor');
title('part a');
legend('2 dimensions with identity covariance matrix','5 dimensions with identity covariance matrix','10 dimensions with identity covariance matrix');
hold off;
toc;

function plotAverageMeanDistance(dim, color)
sampleSize = 1000;
output = zeros(1, sampleSize);
sigma = eye(dim);
for m = 1 : sampleSize
    meanDistance = 0;
    mu = zeros(m,dim);
    if(m>1)
        for i = [1, 100]
                sample = multiVariateNormalDistribution(mu,sigma);
                b = pairwiseDistanceOptimized(sample);
                b = b + diag(Inf(size(diag(b))));
                meanDistance = meanDistance + mean(min(b,[],2));           
        end
    end
    averageMeanDistance = meanDistance/100;
    output(m) = averageMeanDistance;
end
plot(output, color);

function Y = pairwiseDistance(X)
[m,d] = size(X);
P1 = repmat(permute(X, [1 3 2]),[1 m 1]);
P2 = repmat(permute(X, [3 1 2]),[m 1 1]);
diff = P1 - P2;
Y = sqrt(sum(diff.*diff,3));

function Y = pairwiseDistanceOptimized(X)
% This functions returns the matrix containing pairwise euclidean distances
% between the row vectors of input matrix.
[m,d] = size(X);
Y = zeros(m);
for i = 1:m-1
    dsq = zeros(m-i,1);
    for j = 1: d
        dsq = dsq + (X(i,j) - X((i+1):m,j)).^2;
    end
    Y(i,i+1:m) = sqrt(dsq);
    Y(i+1:m,i) = sqrt(dsq);
end

function X = multiVariateNormalDistribution(mu, sigma)
% This functions returns the matrix containing vectors from Multivarite
% Normal Distribution. As the covariance matrix is positive definite we can
% use its square root to generate these vectors.
[m,d] = size(mu);
T = sqrtm(sigma);
X = mu + randn(m,d) * T;