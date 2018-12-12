function gridx=startingGrid(eigv,Beta,N0,Nin,Nfu)

%STARTINGGRID computes the starting grid performing gross support detection
%   GRIDX=STARTINGGRID(EIGV,BETA,{NIN})
%   * EIGV is a set of M population eigenvalues
%   * BETA is the aspect ratio
%   * {N0} is the baseline number of points for support detection. It
%   defaults to 100
%   * {NIN} is the minimum number of points to subdivide the search within 
%   each detected interval. It defaults to 11
%   * {NFU} is the minimum number of points per eigenvalue on the grid. It
%   defaults to 3
%   * GRIDX is the generated grid
%

if nargin<3 || isempty(N0);N0=100;end
if nargin<4 || isempty(Nin);Nin=11;end
if nargin<5 || isempty(Nfu);Nfu=3;end

gpu=isa(eigv,'gpuArray');

%CONSERVATIVE SUPPORT LIMITS
pm=[-1 1]; 
Fact=1+1e-3;%Factor to extend the limits for numerical purposes
BetaLim=(Fact.^pm).*(1+pm.*sqrt(Beta)).^2;

eigv=sort(eigv,1);
M=size(eigv,1);

grAd=gather(bsxfun(@times,BetaLim,eigv));
grDi=(grAd(1:end-1,2,:)<grAd(2:end,1,:));%These are zones we initially guess are not on the support

NS=size(grDi,3);
indBr=cell(1,NS);
NG=zeros(1,NS);
for s=1:NS
    indBr{s}=cat(1,0,find(grDi(:,:,s)),M);%Intervals
    NG(s)=sum(max(round(Nfu*N0*diff(indBr{s})/M),Nin));%Grid sizes
end

%THE GRID IS ADAPTED TO THE DETECTED SUPPORT INTERVALS (UNIFORM AND WITH
%DENSITY GOVERNED BY NUMBER OF POINTS SO AS TO BALANCE RELATIVE WEIGHTS)
%IMPROVED DETECTION OF SMALL COMPONENTS BY OPERATING ON A LOGARITHMIC SCALE
gridx=zeros([max(NG) 1 NS],'like',grAd);%We book memory with precomputed size of the grid
for s=1:NS
    cont=0;
    for n=1:length(indBr{s})-1
        NP=max(round(Nfu*N0*(indBr{s}(n+1)-indBr{s}(n))/M),Nin);
        gridx(cont+1:cont+NP,1,s)=exp(linspace(log(grAd(indBr{s}(n)+1,1,s)),log(grAd(indBr{s}(n+1),2,s)),NP));
        cont=cont+NP;
    end
    if NG(s)<max(NG)%We fill by homogeneously distributing points
        w=diff(gridx(1:NG(s),1,s));
        NP=max(NG)-NG(s);
        subx=gridSubdivide(w,NP);
        gridx(NG(s)+1:max(NG),1,s)=fillGridPoints(gridx(1:NG(s),1,s),subx,NP);
    end
end
if gpu;gridx=gpuArray(gridx);end
gridx=sort(gridx,1);


