function [trees,w] = boosting(data,depth,numberOfTrees)
% Nikhil Kamthe
% 861245635
% 11/15/2016
% CS 229
% PS 6
%
% This method generates decision trees and the corresponding 
% weigths which are then used for boosting decision tree classifier.

[m,d] = size(data);
alpha = ones(m,1);
trees = cell(1,numberOfTrees);
w = [1,numberOfTrees];
for i = 1:numberOfTrees
    train_x = data(:,1:d-1);
    train_y = data(:,d);
    trees{i} = traindtw(train_x,train_y,alpha,depth);
    y_pred = dt(train_x,trees{i});
    indices = find(sign(y_pred) ~= sign(train_y));    
    error = (sum(alpha(indices))/sum(alpha));
    w(i) = log((1-error)/error);
    alpha(indices) = alpha(indices)*exp(w(i));
end 
end