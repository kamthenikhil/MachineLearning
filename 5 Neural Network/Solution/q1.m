function q1(numrep)
% Nikhil Kamthe
% 861245635
% 11/08/2016
% CS 229
% PS 5

data = load('class2d.ascii','-ascii');
x = data(:,1:2);
y = data(:,3);

hiddenLayersVector = [1 2];
hiddenUnitsPerHiddenLayerVector = [1 5 20];
lambdaVector = [0.001 0.01 0.1];

plotCount = 1;

for p = 1:length(hiddenLayersVector)
    hiddenLayers = hiddenLayersVector(p);
    for q = 1:length(hiddenUnitsPerHiddenLayerVector)
        hiddenUnitsPerHiddenLayer = hiddenUnitsPerHiddenLayerVector(q);
        for r = 1:length(hiddenUnitsPerHiddenLayerVector)
            lambda = lambdaVector(r);
            runNeuralNetwork(x,y,hiddenLayers,hiddenUnitsPerHiddenLayer,lambda,numrep,plotCount);
            plotCount = plotCount + 1;
        end
    end
end

end

function runNeuralNetwork(x,y,hiddenLayers,hiddenUnitsPerHiddenLayer,lambda,numrep,plotCount)
% The following method runs trains the neural network. It uses the values
% of the optimal weight matrices to plot the decision boundaries.

figure(plotCount);

[m,d] = size(x);
% w stores weight matrices for all the layers in a form of a cell array. The script
% has been designed to dynamically adjust to changing values of hidden
% layers.
w = cell(1,hiddenLayers+1);

% The experiment is being repeated numrep times. We use the following
% variables to store the optimal values of weights and loss functions.
% loss_final stores the value of minimum total loss so far.
loss_final = Inf;
% w_final stores the weight matrices corresponding to minimum loss.
w_final = w;

for count = 1:numrep 
    stepSize = 1;    
    % Initialize weight matrices.
    for i = 1 : hiddenLayers+1
        if i==1
            w_i = fetchRandomWeightMatrix(d+1,hiddenUnitsPerHiddenLayer);
        elseif i==hiddenLayers+1
            w_i = fetchRandomWeightMatrix(hiddenUnitsPerHiddenLayer+1,1);
        else
            w_i = fetchRandomWeightMatrix(hiddenUnitsPerHiddenLayer+1,hiddenUnitsPerHiddenLayer);
        end
        w{i} = w_i;
    end
    
    w_old = w;
    [f,a_old,z_old] = forwardPropagation(x,w_old,hiddenLayers);
    loss_old = computeLossFuction(y,f,w_old,lambda);
    loss_delta = 1;
    
    % the following variable is used to store the value of previous value
    % of loss. It might seem redundant but it comes handy when 
    loss = loss_old;

    while abs(loss_delta) > abs(10^-8*loss)
        delta = backwardPropagation(y,z_old,w_old,a_old,hiddenLayers);
        w_new = updateWeightMatrices(w_old,delta,z_old,stepSize,lambda);
        [f,a_new,z_new] = forwardPropagation(x,w_new,hiddenLayers);
        loss_new = computeLossFuction(y,f,w_new,lambda);
        loss_delta = loss_new - loss_old;
        if loss_delta < 0
            stepSize = stepSize*1.05;
            loss = loss_old;
            loss_old = loss_new;
            w_old = w_new;
            z_old = z_new;
            a_old = a_new;
        else
            stepSize = stepSize*0.5;
        end
    end

    % Check if the value of the loss fucntion for this iteration is the
    % less than the current minimum. If yes, update the value of final
    % loss fuction (minimum) to value of current loss function. Also update the value
    % final weight vectors to current weight vectors.
    if computeLossFuction(y,f,w_old,lambda)<loss_final
        loss_final = computeLossFuction(y,f,w_old,lambda);
        w_final = w_old;
    end

end

% Use the final weight vectors to plot decision boundaries.
plotclassifier(x,y,@(X) forwardPropagation(X,w_final,hiddenLayers),0.5,0);
title(strcat('layers = ',num2str(hiddenLayers),' hidden = ',num2str(hiddenUnitsPerHiddenLayer),' lambda = ',num2str(lambda)));
end

function w_new = updateWeightMatrices(w_old,delta,z,stepSize,lambda)
% The following method updates the weight matrices using batch gradient
% descent update.

l = length(w_old);
w_new = cell(1,l);
for i = 1:l
    w_new{i} = w_old{i} - stepSize*z{i}'*delta{i} - stepSize*lambda*w_old{i};
end

end

function [f,a,z] = forwardPropagation(x,w,hiddenLayers)
% The following method performs forward propagation on x using the weight
% matrices for each layer. w contains all the weight matrices in its cell
% blocks.

[m,d] = size(x);
a = cell(1,hiddenLayers+1);
z = cell(1,hiddenLayers+2);
z_0 = [ones(m,1) x];
z{1} = z_0;

for i = 1 : hiddenLayers+1
    a_temp = z{i}*w{i};
    a{i} = a_temp;   
    if i == hiddenLayers+1
        z_temp = signoid(a_temp);
    else
        z_temp = [ones(m,1) signoid(a_temp)];
    end    
    z{i+1} = z_temp;
end

f = z{hiddenLayers+2};

end

function delta = backwardPropagation(y,z,w,a,hiddenLayers)
% The following method performs backward propagation on x using the weight
% matrices for each layer. w contains all the weight matrices in its cell
% blocks.

delta_temp = z{hiddenLayers+2} - y;
delta = cell(1,hiddenLayers+1);

delta{hiddenLayers+1} = delta_temp;

for i = hiddenLayers: -1 : 1
    delta_temp = derivativeSignoid(a{i}).*(delta{i+1}*(w{i+1}(2:end,:))');
    delta{i} = delta_temp;
end
end

function l = computeLossFuction(y,f,w,lambda)
% The following method computes the value of loss function.

[m,d] = size(y);
a = sum((y.*log(f) + (1-y).*log(1-f)));
a = -(1/m)*a;
b = 0;
for i = 1:length(w)
    b = b + sum(sum(w{i}.^2,1));
end
b = (lambda*b)/(2*m);
l = a+b;
end

function w = fetchRandomWeightMatrix(m,n)
% The following method returns a random weight matrix of given size from
% the interval [-1, 1].

w = 2.*rand(m,n)-1;
end

function f = signoid(k)
% Computes the value of signoid function for input k.

f = 1./(1+exp(-k));
end

function f = derivativeSignoid(k)
% Computes the value of signoid function for input k.

f = signoid(k).*(1-signoid(k));
end