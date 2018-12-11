function Vq=interp1GPU(X,V,Xq,typ)

%INTERP1GPU performs fully parallel linear interpolation along the first 
%dimension of 2D arrays. The method is based on mapping the second
%dimension into the first one
%   VQ=INTERP1GPU(X,V,XQ,{TYP})
%   * X is an array with the locations of the elements in V
%   * V is an array with the function to be interpolated
%   * XQ is an array with the locations of the interpolated data
%   * TYP is the type of interpolation, 0 for standard interpolation, 1
%   (default) for mapped interpolation
%   * VQ is the interpolated data
%

if nargin<4 || isempty(typ);typ=1;end

N=size(X);
if ~typ
    Vq=Xq;
    for p=1:N(2);Vq(:,p)=interp1(X(:,p),V(:,p),Xq(:,p),'linear',0);end
else
    [X,Xq]=parUnaFun({X,Xq},@double);
    Next=N(1)/(N(1)-1);
    Xor=X(1,:);%Origin
    X=bsxfun(@minus,X,Xor);%Reference to origin
    Xra=X(end,:)*Next;%Range
    XorLi=cumsum(Xra,2);%Origin when mapped to 1D
    XorLi=circshift(XorLi,[0 1]);
    XorLi(1)=0;
    X=bsxfun(@plus,X,XorLi);
    Xq=bsxfun(@plus,Xq,bsxfun(@minus,XorLi,Xor));
    Vq=single(interp1(X(:),V(:),Xq(:),'linear',0));
    Vq=reshape(Vq,[size(Xq,1) N(2)]);
end

