function ps1()
    tic;
    m = 1000;
	result = zeros(m, 1);
    d = 2;
    sigma = eye(d);
    for i = 2:m
        temp = 0;
        for sample = 1:100
            X = randn(i, d)*sigma;
            temp = temp + meanDistance(X);
        end
        result(i,1) = temp/100;
    end
    plot(result);
    toc;
end

function m_d = meanDistance(X)
	T = sqdist(X);
	T(T == 0) = inf;
	T = min(T, [], 2);
	m_d = mean(T);
end

function D = sqdist(X)
    D = squareform(pdist(X));
% 	[m, n] = size(X);
% 	P1 = repmat(permute(X, [1 3 2]), [1 m 1]);
% 	P2 = repmat(permute(X, [3 1 2]), [m 1 1]);
% 	diff = P1 - P2;
% 	D = sqrt(sum(diff.*diff, 3));
end