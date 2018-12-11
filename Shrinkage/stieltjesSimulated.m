function mhv=stieltjesSimulated(esd,ei,di,beta,de,mh)

%STIELTJESSIMULATED computes the Stieltjes transform (ST) for a given 
%empirical sample distribution obtained by simulation
%   MVH=STIELTJESSIMULATED(ESD,EI,{DI},{BETA},{DE},{MH})
%   * ESD is an array of size Mx1xO containing the synthesized eigenvalues
%   of the ESD in the field SIMU
%   * EI is a set of eigenvalues where we want to compute the ST (row 
%   vector)
%   * {BETA} is the shape factor, it is assumed to be lower or equal to 1. 
%   Defaults to 1.
%   * {DI} indicates the convention of the pdf for the ST 
%   (1->X^H*X, 0->X*X^H). Defaults to 1 (generally with beta>1)
%   * {DE} serves to compute derivatives of the ST (defaults to zero for 
%   the transform as such)
%   * {MH} provides precomputed STs at the same derivative order to be used
%   when DI=0
%   * MHV is the ST mhat (DI=1) or vhat (DI=0)
%

if nargin<2 || isempty(di);di=1;end
if nargin<3 || isempty(beta);beta=1;end
if nargin<4 || isempty(de);de=0;end
if nargin<5;mh=[];end

if di==1
    dei=bsxfun(@minus,esd.simu,ei);dei(abs(dei)<1e-4)=1e-4;%x-lambda
    mhv=factorial(de)*mean(dei.^(-(1+de)),1);%sum_k p(x_k)/(x_k-lambda) for de=0    
else
    if isempty(mh);mh=stieltjesSimulated(esd,ei,1,beta,de);end  
    mhv=beta*mh-factorial(de)*((-1)^de)*(1-beta)./(ei.^(1+de));
end
