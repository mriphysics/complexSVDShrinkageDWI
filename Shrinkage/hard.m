function [nsv,nsvd]=hard(sv,beta,cd)

%HARD gives the eigenvalue shrinkage function and its derivative 
%based on a hard thresholding criterion
%   [NSV,NSVD]=HARD(SV,{BETA},{CD})
%   * SV are the singular values to be shrinked
%   * {BETA} is the shape factor, it is assumed to be lower or equal to 1. 
%   Defaults to 1.
%   * {CD} indicates whether to compute the derivative of the shrinkage
%   operator, which could be later used to obtain the SURE (non asymptotic
%   risk), defaults to 0
%   * NSV are the shrinked singular values
%   * NSVD are the derivatives of the shrinked singular values
%

if nargin<2 || isempty(beta);beta=1;end
if nargin<3 || isempty(cd);cd=0;end

th=sqrt(2*(beta+1)+8*beta/(beta+1+sqrt(beta^2+14*beta+1)));
isa=(sv>th);

nsv=sv;
nsv(~isa)=0;

if cd
    nsvd=nsv;
    nsvd(isa)=1;
else
    nsvd=[];
end
