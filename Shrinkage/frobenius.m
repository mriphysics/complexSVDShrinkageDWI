function [nsv,amse]=frobenius(sv,beta,esd,nsvc)

%FROBENIUS gives the eigenvalue shrinkage function and its derivative 
%based on a Frobenius norm criterion
%   [NSV,NSVD]=FROBENIUS(SV,{BETA},{ESD},{NSVC},{NEVERZERO})
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
%   * {NSVC} are a set of shrinked singular values to reference errors 
%   against
%   * NSV are the shrinked singular values
%   * AMSE is an estimate of the asymptotic mean squared error of the 
%estimation
%

if nargin<2 || isempty(beta);beta=1;end
if nargin<3;esd=[];end
if nargin<4;nsvc=[];end

if isempty(esd)%Standard ESD
    isv=(sv>1+sqrt(beta));

    aux=sv(isv).^2-beta-1;
    auxy=sv(isv);

    nsv=sv;
    nsv(isv)=sqrt(aux.^2-4*beta)./auxy;
    nsv(~isv)=0;

    x=operator(sv,beta);
    x2=x(isv).^2;
        
    amse=nsv;
    amse(isv)=x2-((x2.^2-beta).^2)./((x2+beta).*(x2.^2+x2));
    amse=sum(max(amse,0),2);
elseif ~iscell(esd)
    ei=sv.^2;
    if ~isfield(esd,'thre');isv=bsxfun(@gt,ei,esd.simu(end,1,:));else isv=bsxfun(@gt,ei,esd.thre(1,1,:));end
    
    mh=stieltjes(esd,ei);%Conventional Stieltjes
    vh=stieltjes(esd,ei,0,beta,0,mh);%Opposed convention Stieltjes
    Dh=ei.*mh.*vh;%D-transform
    
    mhp=stieltjes(esd,ei,1,beta,1);%Derivative of conventional Stieltjes
    vhp=stieltjes(esd,ei,0,beta,1,mhp);%Derivative of opposed convention Stieltjes
    mvhp=mhp.*vh+mh.*vhp;
    Dhp=mh.*vh+ei.*mvhp;%Derivative of D-transform
    
    nsv=-1./(sv.*Dhp./Dh);
    nsv(~isv)=0;
    nsv(nsv<0)=0;
    
    amse=1./Dh-(1./(sv.*(Dhp./Dh))).^2;
    amse(~isv)=0;  
    amse=sum(abs(max(amse,0)),2);   
else
    O=size(sv,3);
    gpu=isa(sv,'gpuArray');
    sv=gather(sv);
    nsv=sv;
    amse=nsv;
    parfor o=1:O
        svo=sv(:,:,o);
        esdo=esd{o};   
        nsvo=nsv(:,:,o);
        amseo=amse(:,:,o);amseo(:)=0;
        if ~isempty(esdo)
            ei=svo.^2;%SVs to eigs        
            isv=(ei>esdo.thre);    
            nsvo(~isv)=0;        

            if any(isv(:))
                ei=ei(isv);
                mh=stieltjes(esdo,ei);%Conventional Stieltjes
                vh=stieltjes(esdo,ei,0,beta,0,mh);%Opposed convention Stieltjes
                Dh=ei.*mh.*vh;%D-transform

                mhp=stieltjes(esdo,ei,1,beta,1);%Derivative of conventional Stieltjes
                vhp=stieltjes(esdo,ei,0,beta,1,mhp);%Derivative of opposed convention Stieltjes
                mvhp=mhp.*vh+mh.*vhp;
                Dhp=mh.*vh+ei.*mvhp;%Derivative of D-transform        
                nsvo(isv)=-1./(svo(isv).*Dhp./Dh);
                amseo(isv)=1./Dh-(1./(svo(isv).*(Dhp./Dh))).^2;   
            end
        else
            nsvo(:)=0;amseo(:)=0;
        end
        amse(:,:,o)=amseo;
        nsv(:,:,o)=nsvo;
    end  
    if gpu;[amse,nsv]=parUnaFun({amse,nsv},@gpuArray);end
    amse=sum(max(amse,0),2);
end

if ~isempty(nsvc);amse=amse+sum((nsvc-nsv).^2,2);end%Non optimal case

