function nsv=operator(sv,beta,esd)

%OPERATOR gives the eigenvalue shrinkage function and its derivative 
%based on an operator norm criterion
%   [NSV,NSVD]=OPERATOR(SV,{BETA})
%   * SV are the singular values to be shrinked
%   * {BETA} is the shape factor, it is assumed to be lower or equal to 1. 
%   Defaults to 1.
%   * {ESD} is to be introduced when basing the computations on a 
%generalized empirical sample distribution. It is a cell containg a set 
%of structures with an estimate of the empirical spectral distribution of 
%eigenvalues. It should contain the following fields:
%       - ESD.GRID, the grid on which the empirical spectral distribution
%       is defined (row vector)
%       - ESD.DENS, the empirical spectral distribution density (row
%       vector)
%   * {CD} indicates whether to compute the derivative of the shrinkage
%   operator, which could be later used to obtain the SURE (non asymptotic
%   risk), defaults to 0
%   * NSV are the shrinked singular values
%

if nargin<2 || isempty(beta);beta=1;end
if nargin<3;esd=[];end

if isempty(esd)
    isv=(sv>1+sqrt(beta));

    aux=sv(isv).^2-beta-1;
    nsv=sv;
    nsv(isv)=sqrt((aux+sqrt(aux.^2-4*beta))/2);
    nsv(~isv)=0;
elseif ~iscell(esd) && isfield(esd,'simu');
    ei=sv.^2;%SVs to eigs
    isv=bsxfun(@ge,ei,esd.simu(end,1,:));
    
    mh=stieltjesSimulated(esd,ei);%Conventional Stieltjes
    vh=stieltjesSimulated(esd,ei,0,beta,0,mh);%Opposed convention Stieltjes
    Dh=ei.*mh.*vh;%D-transform
    nsv=1./Dh;
    nsv(~isv)=0;
    nsv=sqrt(nsv);%Eigs to SVs      
elseif ~iscell(esd)
    ei=sv.^2;%SVs to eigs
    isv=bsxfun(@ge,ei,esd.thre(1,1,:));
    
    mh=stieltjes(esd,ei);%Conventional Stieltjes
    vh=stieltjes(esd,ei,0,beta,0,mh);%Opposed convention Stieltjes
    Dh=ei.*mh.*vh;%D-transform
    nsv=1./Dh;
    nsv(~isv)=0;
    nsv=sqrt(nsv);%Eigs to SVs     
else
    O=size(sv,3);
    gpu=isa(sv,'gpuArray');
    sv=gather(sv);
    nsv=sv;
    parfor o=1:O
        svo=sv(:,:,o);  
        esdo=esd{o};
        nsvo=nsv(:,:,o);
        if ~isempty(esdo)
            ei=svo.^2;%SVs to eigs  
            isv=(ei>esdo.thre);
            nsvo(~isv)=0;
            if any(isv(:))
                ei=ei(isv);
                mh=stieltjes(esdo,ei);%Conventional Stieltjes
                vh=stieltjes(esdo,ei,0,beta,0,mh);%Opposed convention Stieltjes
                Dh=ei.*mh.*vh;%D-transform
                nsvo(isv)=1./Dh;
            end
        else
            nsvo(:)=0;
        end
        nsv(:,:,o)=nsvo;
    end
    if gpu;nsv=gpuArray(nsv);end
    nsv=sqrt(nsv);%Eigs to SVs
end
