function C=mat2bldiag(X,indBl)

%MAT2BLDIAG   Converts (forwards and backwards) a matrix to a cell-based 
%block diagonal representation. The direction of conversion is automatically
%determined by checking the type of X
%   C=MAT2BLDIAG(X,INDBL)
%   * X is a square matrix / set of blocks arranged in cells
%   * INDBL are the sizes of the blocks
%   * C are the set of blocks arranged in cells / a square matrix
%

di=~iscell(X);%Direction of conversion

indBEnd=cumsum(indBl);
indBSta=[1 indBEnd(1:end-1)+1];
if di
    NC=length(indBl);
    C=cell(1,NC);
    NX=size(X);
    assert(NX(1)==NX(2),'The matrix has to be squared, but it is (%d x %d)',NX(1),NX(2));
    assert(gather(indBEnd(end))==NX(1),'The sum of block sizes (%d) has to match the matrix size (%d)',indBEnd(end),NX(1));    
    for n=1:NC
        indB=indBSta(n):indBEnd(n);
        C{n}=dynInd(X,{indB,indB},1:2);
    end
else
    NC=length(X);
    NX=size(X{1});
    NX(1:2)=gather(indBEnd(end));
    C=zeros(NX,'like',X{1});
    for n=1:NC
        indB=indBSta(n):indBEnd(n);
        C=dynInd(C,{indB,indB},1:2,X{n});
    end
end
