function [w,b] = learnsvm(X,Y,C)
% Nikhil Kamthe
% 861245635
% 10/20/2016
% CS 229
% PS 3

[m,d] = size(X);
% Initializing variables
w = zeros(d,1);
b = 0;
step = 1/(C*m);

% Compute initial value of loss fuction
L_before = computeLoss(X,Y,C,w,b);

% Stop when size of step becomes less than 10^-6/(C*m)
while step >= 10^-6/(C*m)
    % Take 100 steps for every step size.
    for i = 1:100
       [w,b] = performStep(X,Y,C,w,b,step);
    end
    % Compute value of loss function after 100 steps.
    L_after = computeLoss(X,Y,C,w,b);
    % Compare value of loss function before and after taking 100 steps
    % and updating size of step accordingly.
    if L_after < L_before
        % Increase the step size by 5% when objective is improved.
        step = step*105/100;
    else
        % Decrease the step size by 50% when objective is made worse.
        step = step*50/100;
    end
    L_before = L_after;
end

function L = computeLoss(X,Y,C,w,b)
% This fucntion computes the value of loss function for SVM.

[m,d] = size(X);
sum = 0;
for i = 1:m
   f = X(i,:)*w + b;
   if Y(i)*f < 1
       sum = sum + 1 - Y(i)*f;
   end
end
L = C*sum + (w'*w)/2;

function [w,b] = performStep(X,Y,C,w,b,step)
% This function updates value of w and b by using update rules supplied in 
% q1.pdf for gradient descent. 

[m,d] = size(X);
dlw = w;
dlb = 0;
for i = 1:m
   f = X(i,:)*w + b;
   if Y(i)*f < 1
       % In q1.pdf each data point in x is stored as the column vector and
       % w is a column vector. But here each data point in x is a row
       % vector and w is a column vector. Therefore using transpose of
       % C*Y(i)*X(i,:) while updating w.
       dlw = dlw - (C*Y(i)*X(i,:))';
       dlb = dlb - C*Y(i);
   end
end
w = w - step*dlw;
b = b - step*dlb;