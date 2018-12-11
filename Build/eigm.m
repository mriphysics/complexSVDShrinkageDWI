function eigv=eigm(X)

%EIGM   Computes the eigenvalues of a set of Hermitian(!) matrices using 
%multithreaded acceleration
%   EIGV=EIGM(X)
%   * X is the input matrix
%

gpu=isa(X,'gpuArray');

NX=size(X);NX(end+1:3)=1;
[X,NXP]=resSub(X,3:numDims(X));NXP(end+1:3)=1;
X=(X+matfun(@ctranspose,X))/2;%Force the matrix to be Hermitian
X=gather(X);
eigv=dynInd(X,1,2);

if NXP(3)>=8
    parfor o=1:NXP(3);eigv(:,1,o)=eig(X(:,:,o));end
else
    for o=1:NXP(3);eigv(:,1,o)=eig(X(:,:,o));end
end
if gpu;eigv=gpuArray(eigv);end    
eigv=reshape(eigv,[NX(1) 1 NX(3:end)]);

    