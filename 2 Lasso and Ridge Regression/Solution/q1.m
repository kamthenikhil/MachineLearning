function q1()
% Nikhil Kamthe
% 861245635
% 10/08/2016
% CS 229
% PS 2

data = load('comm.txt','-ascii');
[m,d] = size(data);
trainingData = data(1:1000,:);
testingData = data(1001:end,:);
lambdaCount = 100;
lambda = logspace(-6,-1,lambdaCount);

x_train = trainingData(:,1:99);
y_train = trainingData(:,100);
[w,stats] = lasso(x_train,y_train,'Lambda',lambda);
[p,q] = size(y_train);

%part a
Z = bsxfun(@minus,repmat(y_train,[1,lambdaCount]),stats.Intercept) - x_train*w;
mse_train = sum(Z.^2,1)/p;
figure(1);
clf;
semilogx(lambda, mse_train);
hold on;

x_test = testingData(:,1:99);
y_test = testingData(:,100);

% subtracting intercept from y
Z = bsxfun(@minus,repmat(y_test,[1,lambdaCount]),stats.Intercept) - x_test*w;
[p,q] = size(y_test);
mse_test = sum(Z.^2,1)/p;
semilogx(lambda, mse_test);

% part b
figure(2);
clf;
semilogx(lambda, w);
xlabel('\lambda');
ylabel('weights');
title('part b');

% part c
folds = 10;
foldSize = round(m/folds);
k_fold_mse_test = zeros(1,lambdaCount);
for i = 1:folds
    if(i==folds)
        testingData = data((i-1)*foldSize+1:end,:);
    else
        testingData = data((i-1)*foldSize+1:foldSize*i,:);
    end
    
    trainingData = setdiff(data, testingData, 'rows');
    x_train = trainingData(:,1:99);
    y_train = trainingData(:,100);
    [w,stats] = lasso(x_train,y_train,'Lambda',lambda);
    
    x_test = testingData(:,1:99);
    y_test = testingData(:,100);
    Z = bsxfun(@minus,repmat(y_test,[1,lambdaCount]),stats.Intercept) - x_test*w;
    [p,q] = size(y_test);
    k_fold_mse_test = k_fold_mse_test + sum(Z.^2,1)/p;
end

% Take average of errors.
k_fold_mse_test = k_fold_mse_test/folds;

figure(1);
semilogx(lambda, k_fold_mse_test);

xlabel('\lambda');
ylabel('average squared error');
title('part a');
legend('training data','testing data','10-fold cross-validation');
hold off;