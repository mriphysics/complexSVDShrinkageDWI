function mhv=stieltjes(esd,ei,di,beta,de,mh)

%STIELTJES computes the Stieltjes transform (ST) for a given empirical 
%sample  distribution 
%   MVH=STIELTJES(ESD,EI,{DI},{BETA},{DE},{MH})
%   * ESD is a structure with the estimated ESD(s). The structure may
%   contain the following fields:
%       - ESD.GRID, the grid on which the empirical spectral distribution 
%   is defined
%       - ESD.DENS, the empirical spectral distribution density
%       - ESD.THRE, the upper bound on the empirical spectral distribution
%       - ESD.GRIDD, the gradient of the grid locations
%       - ESD.APDF, the accumulated estimated pdf
%       - ESD.SIMU, contains the synthesized eigenvalues obtained by
%       simulation
%   * EI is a set of eigenvalues for which to compute the ST (row vector)
%   * {DI} indicates the convention of the pdf for the ST
%   (1->X^H*X, 0->X*X^H). Defaults to 1 (generally with beta>1)
%   * {BETA} is the shape factor, it is assumed to be lower or equal to 1. 
%   Defaults to 1.
%   * {DE} serves to compute derivatives of the ST (defaults to zero for 
%   the transform as such)
%   * {MH} provides precomputed STs at the same derivative order to be used
%   when DI=0
%   * MHV is the ST mhat (DI=1) or vhat (DI=0)
%

if nargin<3 || isempty(di);di=1;end
if nargin<4 || isempty(beta);beta=1;end
if nargin<5 || isempty(de);de=0;end
if nargin<6;mh=[];end

if di==1
    if isfield(esd,'dens')
        if ~isfield(esd,'gridd');esd.gridd=permute(gradient(permute(esd.grid,[2 1 3])),[2 1 3]);end
        if ~isfield(esd,'apdf');esd.apdf=sum(esd.dens.*esd.gridd,1);end%Integral of the pdf, to normalize for numerical stability
        dei=bsxfun(@minus,esd.grid,ei);dei(abs(dei)<1e-12)=1e-12;%x-lambda
        mhv=factorial(de)*sum(bsxfun(@times,esd.dens.*esd.gridd,(dei).^(-(1+de))),1);%sum_k p(x_k)/(x_k-lambda) for de=0    
        mhv=bsxfun(@rdivide,mhv,esd.apdf);
    end
    if isfield(esd,'simu') && ~isempty(esd.simu)
        dei=bsxfun(@minus,esd.simu,ei);dei(abs(dei)<1e-12)=1e-12;%x-lambda
        if ~isfield(esd,'apdf');mhv=factorial(de)*mean(dei.^(-(1+de)),1);
        else mhv=mhv+bsxfun(@times,1-esd.apdf,factorial(de)*mean(dei.^(-(1+de)),1));%sum_k p(x_k)/(x_k-lambda) for de=0
        end
    end
else
    if isempty(mh);mh=stieltjes(esd,ei,1,beta,de);end  
    mhv=beta*mh-factorial(de)*((-1)^de)*(1-beta)./(ei.^(1+de));
end
