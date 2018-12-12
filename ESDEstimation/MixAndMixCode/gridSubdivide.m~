function subx=gridSubdivide(w,NP)

%GRIDSUBDIVIDE generates the number of subdivisions to be applied to a grid
%to generate NP new points according to per-cell weights W
%   [SUBX]=GRIDSUBDIVIDE(W,NP)
%   * W are the per-cell weights to be equalized
%   * NP are the number of points to be added to the grid
%   * SUBX are the number of per-cell subdivisions
%

N=size(w);N(end+1:2)=1;
N(1)=N(1)+1;

subx=w;subx(:)=1;
imS=(0:N(2)-1);
wCur=w;
for o=1:NP
    [~,imV]=max(wCur,[],1);  
    imV=imV(:)+(N(1)-1)*imS(:);
    subx(imV)=subx(imV)+1;  
    wCur(imV)=w(imV)./subx(imV);
end
