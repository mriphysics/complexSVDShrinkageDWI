function X=diagm(X)

%DIAGM   Extracts the diagonals of multiple square matrices
%   X=DIAGM(X)
%   * X is the input array
%   * X is the output array with the diagonals as a row vector or 
%alternatively a matrix with diagonals given by the input row vector
%

N=size(X);N(end+1:3)=1;ND=length(N);

d=sub2indV([N(2) N(2)],repmat(single((1:N(2))'),[1 2]));
if N(1)==N(2)    
    X=reshape(X,[prod(N(1:2)) prod(N(3:ND))]);
    X=X(d,:);
    X=reshape(X,[1 N(1) N(3:ND)]);
elseif N(1)==1
    Y=repmat(X,[N(2) ones(1,ND-1)]);
    Y(:)=0;
    Y=reshape(Y,[N(2)^2 N(3:ND)]);
    X=reshape(X,[N(2) N(3:ND)]);
    X=dynInd(Y,d,1,X);Y=[];
    X=reshape(X,[N(2) N(2) N(3:ND)]);
else
    error('Matrix sizes do not allow to perform the diagonal operation');
end


