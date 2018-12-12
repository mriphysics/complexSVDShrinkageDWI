function xN=fillGridPoints(x,subx,NP)

%FILLGRIDPOINTS generates a number of points to refine a previous grid on the basis of a given number of subdivisions to the cells of that grid
%   XN=FILLGRIDPOINTS(X,SUBX,NP)
%   * X are the current points on the grid
%   * SUBX are the number of per-cell subdivisions
%   * NP are the number of points to be added to the grid
%   * XN are the new grid points
%

N=size(subx);N(end+1:2)=1;
N(1)=N(1)+1;

xN=zeros([NP N(2)],'like',x);

%THIS IS STRANGELY SLOWLY...
% if N(2)>=8;parforFl=inf;else parforFl=0;end
% 
% parfor(o=1:N(2),parforFl)

if N(2)>=8
    parfor o=1:N(2);xN(:,o)=fillGridPointsBody(subx(:,o),xN(:,o),x(:,o));end
else
    for o=1:N(2);xN(:,o)=fillGridPointsBody(subx(:,o),xN(:,o),x(:,o));end
end

end

function xNo=fillGridPointsBody(subxN,xNo,xo)

indxN=find(subxN~=1);
subxN(subxN==1)=[];
xoi=cat(1,xo(indxN)',xo(indxN+1)');

if ~isempty(subxN)
    xoi=permute(xoi,[1 3 2]);
    c=uniquetol(subxN);
    subxNista=1+cumsum(subxN-1);
    subxNista=[1;subxNista(1:end-1)];

    for q=1:length(c)
        subxNq=c(q);
        vIF=(0:subxNq-2);
        vI=single((vIF+1)/subxNq);       
        qV=find(subxN==c(q));          
        subxNih=bsxfun(@plus,subxNista(qV),vIF);                
        aux=sum(bsxfun(@times,xoi(:,1,qV),[1-vI;vI]),1);
        xNo(subxNih(:))=aux(:);
    end            
end

end