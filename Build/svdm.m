function [U,S,V]=svdm(X)

%SVDM   Performs accelerated multiple svds for wide and square matrices 
%based on a Schur decomposition of the normal matrix X*X'
%   [U,S,V]=SVDM(X)
%   * X is the input matrix
%   * U is the column space (range) of X
%   * S are the singular values of X
%   * V is the row space (range) space of X
%

gpu=isa(X,'gpuArray');
[M,N,O]=size(X);
assert(M<=N,'Only wide and square matrices are allowed but input size is %dx%d',M,N);

Xp=matfun(@ctranspose,X);
U=matfun(@mtimes,X,Xp);
U=(U+matfun(@ctranspose,U))/2;%Force the matrix to be Hermitian  
U=gather(U);S=U;        
parfor o=1:O;[U(:,:,o),S(:,:,o)]=schur(U(:,:,o));end
if gpu;[U,S]=parUnaFun({U,S},@gpuArray);end
S=flip(flip(sqrt(abs(S)),1),2); 
U=flip(U,2);
D=diagm(S);D(D<1e-9)=1;
V=bsxfun(@times,matfun(@mtimes,Xp,U),1./D);


