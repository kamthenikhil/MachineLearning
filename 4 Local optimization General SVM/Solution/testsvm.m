function testsvm()
clf;
tic;
data = load('example2.data','-ascii');
x = data(:,1:2);
y = data(:,3);

explot(x,y);

toc;