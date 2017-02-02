function trees = bagging(data,depth,numberOfTrees)
% Nikhil Kamthe
% 861245635
% 11/15/2016
% CS 229
% PS 6
%
% This method generates decision trees which are then used for
% bagging decision tree classifier. It first generates a bootstrap dataset
% from the input dataset and uses it to get the classifier i.e. tree.

trees = cell(1,numberOfTrees);
for i = 1:numberOfTrees
    bootstrapData = bootstrap(data);
    train_x = bootstrapData(:,1:end-1);
    train_y = bootstrapData(:,end);
    trees{i} = traindt(train_x,train_y,depth);
end

end

function bootstrapData = bootstrap(data)
% The following method generates a bootstrap ddataset using the input
% dataset. It picks random indices from input data (without replacement)
% and uses those indices to create the bootstrap dataset of same size.

[m,d] = size(data);
indices = randi([1 m],m,1);
bootstrapData = data(indices,:);
end