function X=pinvmDamped(X,tol,b)

%PINVMDAMPED   Performs accelerated multiple pinvs based on a Schur decomposition
%of the normal matrix X*X'
%   X=PINVMDAMPED(X,{TOL})
%   * X is the input matrix
%   * {TOL} is a tolerance for singular value truncation. It defaults to 
%   1e-8 (very small)
%   * {B} is the array onto which to perform the pseudoinverse
%   * X is the resulting pseudoinverse
%

if nargin<2;tol=1e-8;end

NX=size(X);NX(end+1:3)=1;
NXr=[NX(1:2) prod(NX(3:end))];
X=reshape(X,NXr);
if nargin>=3
    NB=size(b);NB(end+1:3)=1;
    NBr=[NB(1:2) prod(NB(3:end))];
    b=reshape(b,NBr);
end

NXmin=min(NX(1:2));
%I=tol*bsxfun(@times,eye(NXmin,'like',X),multDimMax(abs(X),1:2).^2);

Xp=permute(conj(X),[2 1 3]);
if NXmin==1
    if NX(1)<=NX(2)
        Xn=sum(abs(X).^2,2);        
        b=bsxfun(@rdivide,b,Xn+tol*Xn+1e-16);            
        X=bsxfun(@times,Xp,b);
    else
        Xn=sum(abs(X).^2,1);
        X=bsxfun(@rdivide,X,Xn+tol*Xn+1e-16);
        X=sum(bsxfun(@times,conj(X),b),1);
    end
    %X=emtimes(Xp,b);       
else            
    I=bsxfun(@times,tol,eye(NXmin,'like',X));        
    %Xp=matfun(@ctranspose,X);    
    if NX(1)<=NX(2)      
        X=emtimes(X,Xp);     
        I=bsxfun(@plus,bsxfun(@plus,bsxfun(@times,I,multDimMax(abs(X),1:2)),1e-16*I),X);
        %I=bsxfun(@plus,bsxfun(@times,I,diagm(diagm(X))),X);
        X=emtimes(Xp,matfun(@mldivide,I,b));
    else
        X=emtimes(Xp,X);       
        I=bsxfun(@plus,bsxfun(@plus,bsxfun(@times,I,multDimMax(abs(X),1:2)),1e-16*I),X);
        %I=bsxfun(@plus,bsxfun(@times,I,diagm(diagm(X))),X);
        X=matfun(@mldivide,I,emtimes(Xp,b));
    end    
end
X=resSub(X,3,NX(3:end));%This instruction may be accelerated