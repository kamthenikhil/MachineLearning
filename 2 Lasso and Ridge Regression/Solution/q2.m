function q2()
% Nikhil Kamthe
% 861245635
% 10/08/2016
% CS 229
% PS 2

fh = figure(1);
clf;
lambdas = [0.001 0.1 10];
for index = 1:3
    lambda = lambdas(index);
    subplot(1,3,index);

    % Generating weight vectors for 1000 samples to plot average function.
    w = zeros(6,1000);
    for i = 1:1000
        [x_train,y_train] = generateSample(10);
        w(:,i) = (x_train'*x_train + lambda*eye(6))\x_train'*y_train;
    end
    
    [x_test,y_test] = generateSample(100);
    y_pred = x_test * w;

    % Plotting 100 of these sample functions in red.
    for i = linspace(1,1000,100)
       plot(x_test(:,2),y_pred(:,round(i)),'r','LineWidth',0.1);
       hold on;
    end
    
    % Plotting average function.
    y_pred_ave = sum(y_pred,2)/1000;
    plot(x_test(:,2),y_pred_ave,'b','LineWidth',2);
    
    % Plotting true function.
    y_true = tan(pi*x_test(:,2)/3)+(x_test(:,2)-0.5).^2;
%     y_true = sin(pi*x_test(:,2));
    plot(x_test(:,2),y_true,'k','LineWidth',2);
    
    % Setting plot properties.
    title(strcat('\lambda=', num2str(lambda)));
    axis([-1 1 -0.5 4.5]);
%     axis([-1 1 -1.5 1.5]);
    hold off;
end
set(fh, 'Position', [0,0,1200,400]);

function [x,y] = generateSample(datasetSize)
% Generates sample data of given input size. x values are drawn uniformly 
% at random from the interval [-1; +1]. y values are computed using f(x).
% A guassian noise of 0.5 is added while computing y values from x.

x = 2.*rand(datasetSize,1)-1;
x = sort(x);
y = zeros(datasetSize,1);
for i = 1:datasetSize
   y(i,1) = tan(pi*x(i)/3)+(x(i)-0.5).^2+0.5*randn();
%    y(i,1) = sin(pi*x(i))+0.5*randn();
end
x = bsxfun(@power,x,0:5);