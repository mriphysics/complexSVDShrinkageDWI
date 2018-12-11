function [nsv,sigma,P]=veraart(sv,beta,mo)

%VERAART  applies singular value thresholding according to the procedure 
%sketched in [1] J Veraart, DS Novikov, D Christiaens, B Ades-aron, 
%J Sijbers, E Fieremans, "Denoising of diffusion MRI using random matrix 
%theory," 142:394-406, 2016
%   [NSV,SIGMA,P]=VERAART(SV,{BETA})
%   * SV are the singular values to be thresholded
%   * {BETA} is the shape factor, it is assumed to be lower or equal to 1. 
%   Defaults to 1.
%   * {MO} is a flag to robustify original estimator by introducing a 
%   modification on the shape factor of the components attributable to 
%   noise. Defaults to 0.
%   * NSV are the thresholded singular values
%   * SIGMA is an estimation of the noise standard deviation
%   * P is the number of components preserved after filtering
%

%INITIALIZATION
if nargin<2 || isempty(beta);beta=1;end
if nargin<3 || isempty(mo);mo=0;end

ei=gather(sv.^2);%Eigenvalues
M=size(sv,2);N=round(M/beta);O=size(sv,3);
gpu=isa(sv,'gpuArray');
sigma=single(ones([1 1 O]));P=single(zeros([1 1 O]));
nsv=sv;

%ESTIMATION
for o=1:O   
    sigma2end=[];
    pmax=M-1;
    for p=0:M-1
        if mo==0;gammap=(M-p)/N;else gammap=(M-p)/(N-p);end
        sigma2p=(ei(1,p+1,o)-ei(1,M,o))/(4*sqrt(gammap));
        sigma2=sum(ei(1,p+1:M,o))/(M-p);        
        if sigma2>=sigma2p
            if isempty(sigma2end) || sigma2-sigma2p<diffsigma%To look for best possible match between estimates diminishing discretization issues
                sigma2end=sigma2;
                diffsigma=sigma2-sigma2p;
                sigma(o)=sqrt(sigma2end);
                P(o)=p;
            else
                break
            end
        end
        if p==pmax;break;end
    end
    nsv(1,P(o)+1:M,o)=0;
end
if gpu;[sigma,P]=parUnaFun({sigma,P},@gpuArray);end