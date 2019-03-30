function [S,U,V]=svdm(X)

%SVDM   Performs accelerated multiple svds for wide and square matrices 
%based on a Schur decomposition of the normal matrix X*X'
%   [S,U,V]=SVDM(X)
%   * X is the input matrix
%   * S are the singular values of X
%   * U is the column space (range) of X
%   * V is the row space (range) space of X
%

gpu=isa(X,'gpuArray');
[M,N,O]=size(X);
%assert(M<=N,'Only wide and square matrices are allowed but input size is %dx%d',M,N);

if nargout>1
    Xp=matfun(@ctranspose,X);
    U=matfun(@mtimes,X,Xp);
    U=(U+matfun(@ctranspose,U))/2;%Force the matrix to be Hermitian  
    U=gather(U);S=U;        
    if O>=8
        parfor o=1:O;[U(:,:,o),S(:,:,o)]=schur(U(:,:,o));end
    else
        for o=1:O;[U(:,:,o),S(:,:,o)]=schur(U(:,:,o));end
    end
    if gpu;[U,S]=parUnaFun({U,S},@gpuArray);end
    S=diagm(sqrt(abs(S)));
    S=flip(S,2);%We use decreasingly sorted singular values
    U=flip(U,2);
    [S,iS]=sort(S,2,'descend');  
    U=indDim(U,iS,2);
    D=S;D(D<1e-9)=1;
    %U=bsxfun(@times,U,1./sqrt(dot(U,U,1)));
    V=bsxfun(@times,matfun(@mtimes,Xp,U),1./D);%PROBLEM WITH SINGLETONS IS THAT THIS MAY DEPART A LOT FROM ORTHONORMALITY!!
else
    X=matfun(@mtimes,X,matfun(@ctranspose,X));
    S=sqrt(abs(eigm(X)));
    S=sort(S,1,'descend');
    S=permute(S,[2 1 3]);
end


