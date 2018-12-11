function [A,sigma,P,amse,amss,sv,nsv]=SVShrinkage(A,shrinkMeth,noiEstMeth,esd,MPmedian)

%SVSHRINKAGE  applies a shrinkage of the singular values of A according to 
%[1] M Gavish, DL Donoho, "Optimal shrinkage of singular values," IEEE 
%Trans Inf Theory, 63(4):2137-2152, 2017, for Frobenius, operator and 
%nuclear shrinkage and median based noise level estimation, [2] M Gavish, 
%DL Donoho, "The optimal hard threshold for singular values is 4/sqrt(3)," 
%IEEE Trans Inf Theory, 60(8):5040-5053, 2014, for hard and soft shrinkage,
%[3] J Veraart, DS Novikov, D Christiaens, B Ades-aron, J Sijbers, 
%E Fieremans, "Denoising of diffusion MRI using random matrix theory," 
%142:394-406, 2016, for expectation based noise level estimation and 
%associated shrinkage
%   [A,SIGMA,P]=SVSHRINKAGE(A,SHRINKMETH,NOIESTMETH,{ESD},{MPMEDIAN})
%   * A is a noisy matrix to be filtered
%   * SHRINKMETH determines the cost function to perform shrinkage. One of 
%   the following values: 'Hard' / 'Soft' / 'Frob' / 'Oper' / 'Nucl' / 
%   'Exp1' / 'Exp2'. Defaults to 'Frob'
%   * NOIESTMETH determines the method to estimate the noise level. One of 
%   the following: 'None' / 'Exp1' / 'Exp2' / 'Medi'. Defaults to 'None' 
%   (noise is standardized)
%   * {ESD} is a cell containing structures with an estimate of the 
%empirical spectral distribution of eigenvalues for each of the pages in 
%matrix A. It should contain the following fields:
%       - ESD.GRID, the grid on which the empirical spectral distribution is
%   defined
%       - ESD.DENS, the empirical spectral distribution density
%       - ESD.THRE, the upper bound on the empirical spectral distribution
%   * {MPMEDIAN} is the median of the Marcenko-Pastur distribution
%   * A is the filtered matrix
%   * SIGMA is an estimation of the noise standard deviation
%   * P is the effective number of components preserved after filtering
%   * AMSE is an estimate of the asymptotic mean squared error of the 
%estimation
%   * AMSS is an estimate of the mean square of the signal
%   * SV are the observed singular values (generally returned only for 
%visualization purposes)
%   * NSV are the estimated singular values (generally returned only for 
%visualization purposes)
%

%DEFAULT VALUES AND INITIALIZATION
if nargin<2 || isempty(shrinkMeth);shrinkMeth='Frob';end
if nargin<3 || isempty(noiEstMeth);noiEstMeth='None';end
if nargin<4;esd=[];end
if nargin<5;MPmedian=[];end

assert(numDims(A)<=3,'Number of dimensions has to be equal or lower than 3 and it is %d',numDims(A));
[M,N,O]=size(A);beta=M/N;
assert(beta<=1,'Shrinkage assumes M<=N (horizontal matrices) and size is %dx%d',M,N);
comp=~isreal(A(1));
gpu=isa(A,'gpuArray');

%SV DECOMPOSITION
[U,E,V]=svdm(A/sqrt(N*(1+comp)));
sv=abs(diagm(E));%Singular values
sv(sv<1e-6)=1e-6;

%NOISE ESTIMATION
if strcmp(shrinkMeth,'Exp1');noiEstMeth='Exp1';end%Forced, as they are linked
if strcmp(shrinkMeth,'Exp2');noiEstMeth='Exp2';end%Forced, as they are linked
if strcmp(noiEstMeth,'Exp1')
    [nsv,sigma,P]=veraart(sv,beta,0);
elseif strcmp(noiEstMeth,'Exp2')
    [nsv,sigma,P]=veraart(sv,beta,1);
elseif strcmp(noiEstMeth,'Medi')%This uses the code provided in Gavish17    
    if isempty(MPmedian);MPmedian=percMarcenkoPastur(beta);end
    sigma = median(sv,2)/sqrt(MPmedian);
elseif strcmp(noiEstMeth,'None')
    sigma=single(ones([1 1 O]));
    if gpu;sigma=gpuArray(sigma);end
else
    error('Unsupported noise estimation method: %s',noiEstMeth);
end

%SHRINKAGE
if ~ismember(shrinkMeth,{'Exp1','Exp2'})        
    if ismember(noiEstMeth,{'Medi','Exp1','Exp2'})
        sv=bsxfun(@rdivide,sv,sigma);
        [nsv,amse]=generalShrinkage(sv,beta,shrinkMeth);
    else
        [nsv,amse]=generalShrinkage(sv,beta,shrinkMeth,esd);
    end    
    P=sum(nsv~=0,2);%Rank estimation as the number of components that emerge from the bulk        
    if ~isempty(esd)
        sv=bsxfun(@times,sv,sigma);
        nsv=bsxfun(@times,nsv,sigma);
    end        
    if ~strcmp(shrinkMeth,'Frob') || ismember(noiEstMeth,{'Medi','Exp1','Exp2'});[~,amse]=generalShrinkage(sv,beta,'Frob',esd,nsv);end
else
    if isempty(esd)
        sv=bsxfun(@rdivide,sv,sigma);
        nsv=bsxfun(@rdivide,nsv,sigma);
    end
    [~,amse]=generalShrinkage(sv,beta,'Frob',esd,nsv);
end
if isempty(esd)
   nsv=bsxfun(@times,nsv,sigma);
   amse=bsxfun(@times,amse,sigma.^2);
end
amss=sum(abs(nsv).^2,2);

%SV SYNTHESIS
nE=diagm(nsv);
nE=nE*sqrt(N*(1+comp));
amse=amse*(1+comp)/M;
amss=amss*(1+comp)/M;
A=matfun(@mtimes,matfun(@mtimes,U,nE),matfun(@ctranspose,V));
