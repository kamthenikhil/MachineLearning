function parta()
% Nikhil Kamthe
% 861245635
% 11/15/2016
% CS 229
% PS 6

tic;
data = load('class2d.ascii','-ascii');
x = data(:,1:2);
y = data(:,3);
depths = 1:3;
numberOfTreesArray = floor(logspace(0,3,4));
figureCount = 1;
for depth = depths
    figure(figureCount);
    subplotCount = 1;
    % To save the time, the trees are generated for the maximum number of
    % trees (in this case 1000). Then depending on the number of trees for
    % the current iteration we compute the predicted values using only
    % those many number of trees.
    [trees] = bagging(data,depth,numberOfTreesArray(end));
    for numberOfTrees = numberOfTreesArray
        subplot(2,2,subplotCount);
        plotclassifier(x,y,@(X) predictBagging(X,trees,numberOfTrees),0.5,0);
        subplotCount = subplotCount + 1;
        t = title(strcat('depth=',num2str(depth),' &',' number of trees=',num2str(numberOfTrees)));
        set(t, 'FontSize', 10);
    end
    figureCount = figureCount + 1;
end
toc;
end

function y = predictBagging(x,trees,numberOfTrees)
% This mothod uses the trees generated by the bagging function to
% predict the output for input data. It uses numberOfTrees param to use
% only a part of trees to predict the output.

y = zeros(length(x),1);
for i = 1:numberOfTrees
    y = y + dt(x,trees{i});
end
y = y/numberOfTrees;
y(y>=0) = 1;
y(y<0) = -1;
end