function [alpha,b] = learnsvm(X,Y,C,kernel)
% [alpha,b] = learnsvm(X,Y,C,kernel)
% X is an m-by-n matrix of m data points (each of n dimensions)
% Y is an m-by-1 vector of labels (+1 or -1)
% C is a positive scalar.  Larger C means smaller margins, but more
%   points on the correct side of the margin
% kernel is a function of two arguments
%      if A is r-by-n and B is s-by-n
%      kernel(A,B) is r-by-s in which the i,j element is
%        the kernel applied to A(i,:) and B(j,:)
%      Thus, the simple "linear kernel" is the function
%           @(A,B) A*B'
%      and a polynomial kernel of degree d (and constant c) is
%           @(A,B) (A*B' + c).^d

	tol = 1e-3;
	K = kernel(X,X);
	n = size(Y,1);
	G = K.*(Y*Y');
	alpha = zeros(n,1);
	b = 0;
	c=1;
	nch = 0;
	exall = 1;
	while nch>0 | exall
		nch = 0;
		if (exall)
			for i=1:length(alpha)
				nch = nch + picksecond(i);
			end
		else
			for i=1:length(alpha)
				if alpha(i)<C && alpha(i)>0
					nch = nch + picksecond(i);
				end
			end
		end
		if exall
			exall=0;
		elseif nch==0
			exall=1;
		end
	end

	function nch = picksecond(i)
		E = ((K*(alpha.*Y)) + b) - Y;
		l = E.*Y;
		nch = 0;
		if (l(i) < -tol && alpha(i)<C) || (l(i) > tol && alpha(i)>0)
			if sum(alpha>0 & alpha<C) > 1
				%disp('a');
				[tmp,j] = max(abs(E-E(i)));
				nch = update(i,j);
			end
			if ~nch
				%disp('b');
				jdel = randi(length(alpha),1);
				for jj=1:length(alpha)
					j = mod(jj+jdel,length(alpha))+1;
					if j~=i && alpha(j)<C && alpha(j)>0
						if update(i,j)
							nch = 1;
							break;
						end
					end
				end
			end
			if ~nch
				%disp('c');
				jdel = randi(length(alpha),1);
				for jj=1:length(alpha)
					j = mod(jj+jdel,length(alpha))+1;
					if j~=i
						if update(i,j)
							nch = 1;
							break;
						end
					end
				end
			end
		end
	end

	function nch = update(i,j)
		% write this function!
		% note that you have access to the variables from above
		% including K, G, Y, and alpha

		% this function needs to return whether it made a change
		% to do this, check to see that the change in alpha(i)
		% was greater than 1e-10 times (the sum of the old and 
		% new values plus 1e-10)
		% if it was, return 1 (you made a change!)
		% if it was not, return 0 (no real change)
		%
		% if there is a change, update b by the following statement        
        
        % Crating vectors containing indices for convenience
        selectedIndices = [i j];
        excludedIndices = 1:length(alpha);
        excludedIndices(selectedIndices) = [];
        
        q = G(excludedIndices, selectedIndices);
        a_tilde = alpha(excludedIndices);
        
        h = Y(i)*Y(j);
        [L,U] = fetchBounds(h,alpha(i),alpha(j));        
        if (G(i,i) - h*G(i,j) - h*G(j,i) + G(j,j)) > 0
            % if its a maxima.
            alpha_i_new = ((1-h)-[1 -h]*q'*a_tilde-(alpha(j)+h*alpha(i))*(G(i,j)-h*G(j,j)))/(G(i,i)-h*G(j,i)-h*G(i,j)+G(j,j));
            if alpha_i_new<L
                alpha_i_new = L;
            elseif alpha_i_new>U
                alpha_i_new = U;
            end
        else
            alpha_copy = alpha;
            alpha_copy(i) = L;
            alpha_copy(j) = alpha(j)+h*(alpha(i)-L);
            Lobj = computeObjective(alpha_copy);
            alpha_copy(i) = U;
            alpha_copy(j) = alpha(j)+h*(alpha(i)-U);
            UObj = computeObjective(alpha_copy);
            if Lobj > UObj
                alpha_i_new = L;
            else
                alpha_i_new = U;
            end
        end
        if abs(alpha(i)-alpha_i_new) > 10^-10*(alpha(i)+alpha_i_new+10^-10)
            alpha_j_new = alpha(j)+h*(alpha(i)-alpha_i_new);
            alpha(i) = alpha_i_new;
            alpha(j) = alpha_j_new;
            b = findb(K*(alpha.*Y),Y);
            nch = 1;
        else
            nch = 0;
        end
    end
    
    function [L,U] = fetchBounds(h,a1,a2)
        % This function is used to generate the bounds for new alpha.
        if h>0
            L = max(0,a1+a2-C);
            U = min(C,a1+a2);
        else
            L = max(0,a1-a2);
            U = min(C,C+a1-a2);
        end
    end

    function objective = computeObjective(a)
        % This function is used to compute the value of objective function
        % for a given value of alpha.
        objective = -1/2*(a'*G*a) + a'*ones(length(a),1);
    end
end

function b = findb(f,Y)

	f(Y>0) = f(Y>0)-1;
	f(Y<0) = f(Y<0)+1;
	[f,i] = sort(f);
	Y = Y(i);
	diffY = (1:(length(Y)+1))'-ceil((length(Y)+1)/2);
	difff = [f;0] - [0;f];
	diff = diffY.*difff;
	v = cumsum(diff);
	v = v(2:end-1);
	[bestv,ii] = min(v);
	if v(ii+1)==bestv
		b = -(f(ii+1)+f(ii+2))/2;
	else
		b = -f(ii+1);
	end
end
