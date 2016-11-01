function q3()
% Nikhil Kamthe
% 861245635
% 10/20/2016
% CS 229
% PS 3

tic;
clf;
% Loading traing and testing datasets.
x_train = load('spamtrainX.data','-ascii');
y_train = load('spamtrainY.data','-ascii');
x_test = load('spamtestX.data','-ascii');
y_test = load('spamtestY.data','-ascii');

% Declaring a sample size for C.
sampleSize = 10;
C_array = logspace(-3,3,sampleSize);

% Creating vectors to store error rates for cross validation and testing
% data. These vectors are then used to plot the error rates.
error_rate_cv = zeros(1,sampleSize);
error_rate_testing_data = zeros(1,sampleSize);

% Computing error rates for different values of C
for index = 1: length(C_array)
    % error rate 3 fold cross validation
    [m,d] = size(x_train);
    folds = 3;
    foldSize = round(m/folds);
    counter = 0;
    for i = 1:folds
        % The following code generates indices for cross validation.
        if i==1
            test_indices = 1:foldSize;
            train_indices = foldSize+1:m;
        elseif i==folds
            test_indices = (i-1)*foldSize+1:m;
            train_indices = 1:(i-1)*foldSize;
        else
            test_indices = (i-1)*foldSize+1:foldSize*i;
            train_indices = [1:(i-1)*foldSize,foldSize*i+1:m];
        end
        % Creating cross validation training and testing dataset using
        % indices created above.
        X_train_cv = x_train(train_indices,:);
        Y_train_cv = y_train(train_indices,:);
        X_test_cv = x_train(test_indices,:);
        Y_test_cv = y_train(test_indices,:);
        % Updating counter for wrong predictions.
        counter = counter + analyze(X_train_cv,Y_train_cv,X_test_cv,Y_test_cv,C_array(index));
    end
    % Computing error rate for cross validation
    error_rate_cv(index) = (counter/m)*100;

    % error rate testing data
    [m,d] = size(y_test);
    counter = analyze(x_train,y_train,x_test,y_test,C_array(index));
    % Computing error rate for testing data.
    error_rate_testing_data(index) = (counter/m)*100;
end

% Plotting error rate for cross validation vs C
semilogx(C_array,error_rate_cv,'b');
hold on;
% Plotting error rate for testing data vs C
semilogx(C_array,error_rate_testing_data,'g');
xlabel('C Value');
ylabel('Error Rate');
title('Q3: SVM Application');
legend('Cross Validation','Testing Data');
hold off;

toc;

function counter = analyze(X_train,Y_train,X_test,Y_test,C)
% The following fuction builds a SVM model using the training dataset.
% It then uses this model to predict values for test dataset.
% The function returns a counter of number of mismatches in actual and
% predicted value of y.

counter = 0;
[w,b] = qplearnsvm(X_train,Y_train,C);
y_pred = X_test * w + b;
[m,d] = size(y_pred);
for i = 1:m
    if Y_test(i)*y_pred(i) <= 0
        counter = counter + 1;
    end
end