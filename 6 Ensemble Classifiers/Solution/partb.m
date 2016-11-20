function partb()
% Nikhil Kamthe
% 861245635
% 11/15/2016
% CS 229
% PS 6

tic;
data = load('class2d.ascii','-ascii');
x = data(:,1:2);
y = data(:,3);
depths = [1 2 3];
numberOfTreesArray = [1 10 100 1000];
figureCount = 1;
for depth = depths
    figure(figureCount);
    subplotCount = 1;
    % To save the time, the trees and the corresponging weights are 
    % generated for the maximum number of trees (in this case 1000).
    % Then depending on the number of trees for the current iteration 
    % we compute the predicted values using only those many number of trees.
    [trees,w] = boosting(data,depth,numberOfTreesArray(end));
    for numberOfTrees = numberOfTreesArray
        subplot(2,2,subplotCount);
        plotclassifier(x,y,@(X) predictBoosting(X,trees,w,numberOfTrees),0.5,0);
        subplotCount = subplotCount + 1;
        t = title(strcat('depth=',num2str(depth),' &',' number of trees=',num2str(numberOfTrees)));
        set(t, 'FontSize', 10);
    end
    figureCount = figureCount + 1;
end
toc;
end
 
function y = predictBoosting(x,trees,w,numberOfTrees)
% This method uses the trees and the corresponding weights generated
% by the boosting function to predict the output for input data. It uses 
% numberOfTrees param to use only a part of trees to predict the output.

y = zeros(length(x),1);
for i = 1:numberOfTrees
    y = y + w(i)*dt(x,trees{i});
end
y(y>=0) = 1;
y(y<0) = -1;
end