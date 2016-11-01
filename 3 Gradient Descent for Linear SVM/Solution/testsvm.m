function testsvm()
clf;
X = load('simple2X.data','-ascii');
Y = load('simple2Y.data','-ascii');
hold on;
[m,d] = size(X);
for i = 1:m
    if Y(i) > 0
        plot(X(i,1),X(i,2),'gx');
    else
        plot(X(i,1),X(i,2),'rx');
    end
end
C = 1000;
[w,b] = learnsvm(X,Y,C);
drawline(w,b);
hold off;