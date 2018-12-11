function [nsv,amse]=generalShrinkage(sv,beta,shrinkMeth,esd,nsv)

%GENERALSHRINKAGE is based on the general_spiked_estimate.m function from 
%the Spectrode package, [1] E Dobriban, "Efficient computation of limit
%spectra of sample covariance matrices," Random Matrices: Theory Appl, 
%4(4):1550019-36 p, Oct 2015, using the implementation at 
%https://github.com/dobriban/eigenedge. It also performs standard shrinkage
%when ESD is absent or empty
%   NSV=GENERALSHRINKAGE(SV,{BETA},{SHRINKMETH},{ESD},{NSV})
%   * SV are the singular values to be shrinked
%   * {BETA} is the shape factor, it is assumed to be lower or equal to 1. 
%   Defaults to 1.
%   * {SHRINKMETH} determines the cost function to perform shrinkage. One of 
%   the following values: 'Hard' / 'Soft' / 'Frob' / 'Oper' / 'Nucl'. 
%   Defaults to 'Frob'
%   * {ESD} is a cell containing structures with an estimate of the 
%empirical spectral distribution of eigenvalues for each of the pages in 
%matrix A. It should contain the following fields:
%       - ESD.GR, the grid on which the empirical spectral distribution is
%   defined
%       - ESD.DENS, the empirical spectral distribution density
%       - ESD.TH, the upper bound on the empirical spectral distribution
%   * {NSV} are the shrinked singular values (to calculate the amse of 
%   non-optimized procedures)
%   * NSV are the shrinked singular values
%   * AMSE is an estimate of the asymptotic mean squared error (only given
%   for Frobenius shrinkage)
%

if nargin<2 || isempty(beta);beta=1;end
if nargin<3 || isempty(shrinkMeth);shrinkMeth='Frob';end
if nargin<4;esd=[];end
if nargin<5;nsv=[];end

amse=[];
if strcmp(shrinkMeth,'Frob');[nsv,amse]=frobenius(sv,beta,esd,nsv);
elseif strcmp(shrinkMeth,'Oper');nsv=operator(sv,beta,esd);
elseif isempty(esd) && strcmp(shrinkMeth,'Hard');nsv=hard(sv,beta);
elseif isempty(esd) && strcmp(shrinkMeth,'Soft');nsv=max(nsv-(1+sqrt(beta)),0);
else error('Unsupported shrinkage method: %s',shrinkMeth);
end
