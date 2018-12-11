function [xh,sh,ph,amseh,Gamma]=patchSVShrinkage(x,dimsM,sp,parR,cov)

%PATCHSVSHRINKAGE   Performs a shrinkage of the SVD over patchs
%extracted from an array and integrates back the information
%   [X,SEST,PEST]=PATCHSVSHRINKAGE(X,DIMSM,{SP},{PARR},{PARV},{COV})
%   * X is the array on which to perform the patch-based SVD shrinkage
%   * DIMSM is the last dimension of X along which to build the rows of the
%   matrices
%   * {SP} is the spacing used to build local patches along the row 
%dimensions. It defaults to 1
%   * {PARR} are the parameters to build the matrices from the pathes
%   * {COV} is a structure containing
%       - COV.LAMBDA, the covariance matrix of noise. If noise is spatially
%       incorrelated, it corresponds to a map of the noise variance,
%       otherwise, first two dimensions giving the correlations along,
%       typically the phase encoding direction in the data and third 
%       dimension lexicographically indexing any other dimension in the 
%       data. Fourth dimension indexes the encoding type a.
%       - COV.DIMSPE, an array of 1xA, with the correlated dimension for
%       each encoding type 1<=a<=A.
%       - COV.DIMSNOPE and array of size (ND-1)xA, with non correlated
%       dimensions.
%   * XH is the recovered array
%   * SH is the recovered std of noise
%   * PH is the recovered number of components
%   * AMSEH is an estimate of the asymptotic mean square error
%   * GAMMA is an estimate of the patch size
%

%DEFAULT VALUES AND DIMENSIONS
onM=ones(1,dimsM);
if nargin<3 || isempty(sp);sp=onM;end
if nargin<4 || isempty(parR)
    parR.useRatioAMSE=1;%To compute the AMSE ratio (1), AMSE (0) or both (2)
    parR.ESDMeth=0;%0->Use simulation / 1-> Use SPECTRODE / 2-> Use MIXANDMIX / 3-> Force computation of simulated spectra (to get AMSE estimates when using noise estimates instead of noise modeling)
    parR.ESDTol=1e-2;%Tolerance for ESD computations (role depends on the method)
    parR.Gamma=0.85;%Random matrix aspect ratio, set of values to estimate
    parR.Subsampling=1;%Subsampling factor for patch construction
    parR.NoiEstMeth='None';%Noise estimation method. One of the following: 'None' / 'Exp1' / 'Exp2' / 'Medi'
    parR.ShrinkMeth='Frob';%Shrinkage method. One of the following: 'Hard' / 'Soft' / 'Frob' / 'Oper' / 'Nucl' / 'Exp1' / 'Exp2'
    parR.NR=1;%Number of realizations of random matrix simulations
    parR.Verbosity=0;%To plot debugging information
    parR.DirInd=0;%To use accelerate by using the independence of covariances along a given direction
end
if nargin<5;cov=[];end

gpu=isa(x,'gpuArray');
if gpu;dev=gpuDevice;end
NX=size(x);NX(end+1:dimsM)=1;ND=length(NX);
N=prod(NX(dimsM+1:ND));%In this implementation the number of columns comprises all the elements in dimsN. This may be generalized in the future 
if length(parR.Subsampling)==1;parR.Subsampling=parR.Subsampling*onM;end

%ARRANGE INPUT DATA AND BOOK MEMORY FOR RECOVERED DATA
x=reshape(x,[prod(NX(1:dimsM)) prod(NX(dimsM+1:ND))]);%Arrange in matrix form
xh=x;xh(:)=0;%Recovered data
sh=xh(:,1);ph=sh;amseh=sh;amssh=sh;wh=sh;

%PATCH STRUCTURE
NGR=ceil(((N*prod(sp)).^(1/dimsM))./sp);
NGD=2*NGR+1;

%DEFINE THE STRUCTURE FOR SAMPLING THE SPECTRUM TO ESTIMATE BEST parR.Gamma
Nsweeps=length(parR.Gamma)+1;
amseGamma=inf(2,Nsweeps);
if gpu;amseGamma=gpuArray(amseGamma);end
if length(parR.Gamma)==1
    Nsweeps=1;
    indBestGamma=1;
end
for ns=1:Nsweeps
    if parR.Verbosity>=3
        if gpu;wait(dev);end
        tsta=tic;
    end
    %EMPIRICAL SAMPLE DENSITY
    if ns~=Nsweeps;Gamma=parR.Gamma(ns);else Gamma=parR.Gamma(indBestGamma);end   
    if parR.Verbosity>=2
        if ns~=Nsweeps;fprintf('Testing candidate Gamma=%.2f\n',Gamma);else fprintf('Denoising using Gamma=%.2f\n',Gamma);end
    end
    M=round(N*Gamma);

    %WEIGHTS OF THE PATCH STRUCTURE
    xx=generateGrid(NGD,gpu,NGD,ceil((NGD+1)/2));
    r=xx{1}(1);r(1)=double(0);
    for n=1:dimsM;r=bsxfun(@plus,r,(xx{n}*sp(n)).^2);end
    [~,ir]=sort(r(:));
    irM=ir(1:M);
    irs=ind2subV(NGD,irM);
    for n=1:dimsM;xx{n}=xx{n}(irs(:,n));end     
    w=spaNeigh(xx,NGR);    
    w=w(irM)/sum(w(:));
    
    %TO FORM MATRIX BLOCKS ALONG THE DIRECTION OF INDEPENDENCE
    if parR.DirInd~=0
        [vz,iz]=sort(irs(:,3));
        [~,iavz,~]=unique(vz);
        iavz=[iavz' M+1];    
        indbldiag=diff(iavz);    
        w=w(iz);
        irs=irs(iz,:);
        %irM=irM(iz);%Not used
    else
        indbldiag=[];
    end

    if ns~=Nsweeps;Subsampling=parR.Subsampling*parR.DrawFact;else Subsampling=parR.Subsampling;end

    %PATCH CENTERS
    cp=generateGrid(ceil(NX(1:dimsM)./Subsampling),0,Subsampling.*ceil(NX(1:dimsM)./Subsampling),onM);
    icp=cp{1}(1);icp=1;    
    for n=1:dimsM;icp=bsxfun(@plus,icp,cp{n}*prod(NX(1:n-1)));end
    icps=ind2subV(NX(1:dimsM),icp); 
        
    %PATCH INDEXES
    O=size(icps,1);    
    icps=reshape(icps,[O 1 dimsM]);
    irs=reshape(irs,[1 M dimsM]);
    icprs=bsxfun(@plus,icps,irs);
    icprs=reshape(icprs,[O*M dimsM]);
    for n=1:dimsM;icprs(:,n)=mod(icprs(:,n)-1,NX(n))+1;end    
    icpr=sub2indV(NX(1:dimsM),icprs);
    icpr=gather(reshape(icpr,[O M]));
    
    %INDEXES TO COVARIANCE
    if ~isempty(cov) && isfield(cov,'Lambda')    
        NC=size(cov.Lambda);NC(end+1:4)=1;
        cov.Lambda=reshape(cov.Lambda,[prod(NC(1:3)) NC(4)]);        
        icprC=[];
        if any(cov.dimPE~=0)%WITH CORRELATIONS, AS IN PARTIAL FOURIER, DIMENSIONS 1-2 CORRELATED DIMENSION, DIMENSION 3, OTHER DIMENSIONS 
            for s=1:NC(4)%NOTE FOURTH DIMENSION CORRESPONDS TO NUMBER OF ENCODINGS, A IN EC.(12) OF THE PAPER
                icprsC=gather(horzcat(icprs(:,cov.dimPE(s)),sub2indV(NX(cov.dimNoPE(:,s)),icprs(:,cov.dimNoPE(:,s)))));%FIRST INDEXES TO CORRELATED DIMENSION, THEN INDEXES TO THE OTHER DIMENSIONS
                if isempty(icprC);icprC=icprsC;else icprC=cat(4,icprC,icprsC);end
            end;icprsC=[];
            icprC=reshape(icprC,[O M 2 NC(4)]);
        else%WITHOUT CORRELATIONS, DIMENSIONS OF COV.C CORRESPOND TO VOXELS IN THE IMAGE
            for s=1:NC(4)
                icprsC=gather(sub2indV(NC(1:3),icprs));
                if isempty(icprC);icprC=icprsC;else icprC=cat(4,icprC,icprsC);end
            end;icprsC=[];
            icprC=reshape(icprC,[O M 1 NC(4)]);
        end
    else
        NC=ones(1,4);
    end
    icprs=[];

    %PRECOMPUTE MARCENKO-PASTUR MEDIAN
    if strcmp(parR.NoiEstMeth,'Medi');MPmedian=percMarcenkoPastur(Gamma);else MPmedian=[];end                    
    
    %SVD SHRINKAGE OVER THE PATCHES    
    if parR.Verbosity>=3;fprintf('Number of patches: %d\n',O);end
    if parR.ESDMeth==2;bS=ceil(6400/(NC(4)*M));else bS=ceil(400/NC(4));end%MODIFICATIONS MAY BE REQUIRED FOR DIFFERENT COMPUTING PLATFORMS
    amseGamma(:,ns)=0;
    contO=0;
    if parR.Verbosity>=3
        if gpu;wait(dev);end
        fprintf('Time building patches: %.3f s\n',toc(tsta));
    end                
    for o=1:bS:O;vO=o:min(o+bS-1,O);%ITERATING ALONG THE PATCHES
        contO=contO+1;
        if ~isempty(vO)
            %PATCH EXTRACTION
            L=length(vO);%NUMBER OF PATCHES TO BE PROCESSED IN EACH CHUNK
            xoh=x(icpr(vO(1),:),:);xoh=repmat(xoh,[1 1 L]);
            for l=2:L
                xoh(:,:,l)=x(icpr(vO(l),:),:);
            end               

            %COMPUTATION OF THE COVARIANCE OVER THE LOCAL PATCHES
            if ~isempty(cov) && isfield(cov,'Lambda')
                if parR.Verbosity>=3
                    if gpu;wait(dev);end
                    tsta=tic;
                end
                if any(cov.dimPE~=0)%WITH CORRELATIONS                    
                    Lambda=eye(M,'like',real(x));
                    Lambda=repmat(Lambda,[1 1 L NC(4)]);Lambda=reshape(Lambda,[M*M 1 L NC(4)]);
                    for s=1:NC(4)
                        %Next four lines could be moved outside
                        MA=icprC(vO(1),:,2,s);
                        if gpu;MA=gpuArray(MA);end
                        MA=repmat(MA,[M 1]);
                        MB=MA';
                        [vFl,vFr]=find(MB==MA);
                        vF=sub2indV([M M],[vFl vFr]);
                        for l=1:L
                            indCAux=sub2indV(NC(1:3),vertcat(icprC(vO(l),vFl,1,s),icprC(vO(l),vFr,1,s),icprC(vO(l),vFl,2,s))');
                            if gpu;Lambda(vF,1,l,s)=gpuArray(cov.Lambda(indCAux,s));else Lambda(vF,1,l,s)=cov.Lambda(indCAux,s);end
                        end
                    end
                    Lambda=reshape(Lambda,[M M L NC(4)]);
                else%WITHOUT CORRELATIONS               
                    Lambda=zeros([M 1 L NC(4)],'like',real(x));
                    for s=1:NC(4)
                        for l=1:L
                            if gpu;Lambda(:,1,l,s)=gpuArray(cov.Lambda(icprC(vO(l),:,1,s),s));else Lambda(:,1,l,s)=cov.Lambda(icprC(vO(l),:,1,s),s);end
                        end
                    end
                end  
                irs=squeeze(irs);                     
                if parR.Verbosity>=3
                    if gpu;wait(dev);end
                    fprintf('Time computing the covariance: %.3f s\n',toc(tsta));
                end    

                %ESTIMATING THE ESD                    
                if parR.Verbosity>=3
                    if gpu;wait(dev);end
                    tsta=tic;
                end                                              
                if strcmp(parR.NoiEstMeth,'None') || parR.ESDMeth==3
                    if parR.ESDMeth==2;esd=ESDMixAndMix(Lambda,N,parR.ESDTol,2,[],indbldiag);
                    elseif parR.ESDMeth==1 && size(Lambda,4)==1;esd=ESDSpectrode(Lambda,M/N,parR.ESDTol);
                    else esd=ESDSimulated(Lambda,N,[],parR.NR);
                    end
                else
                    esd=[];
                end                
                if parR.Verbosity>=3
                    if gpu;wait(dev);end
                    fprintf('Time estimating the ESD: %.3f s\n',toc(tsta));
                end                    
            end 

            %SHRINKING
            if parR.Verbosity>=3
                if gpu;wait(dev);end
                tsta=tic;
            end
            [xoh,soh,poh,amseoh,amssoh]=SVShrinkage(xoh,parR.ShrinkMeth,parR.NoiEstMeth,esd,MPmedian);                
            if parR.Verbosity>=3
                if gpu;wait(dev);end
                fprintf('Time shrinking: %.3f s\n',toc(tsta));
            end
            if parR.Verbosity>=3;fprintf('Estimated AMSE: %.3f\n',mean(amseoh(:)));end

            %ASSEMBLING BLOCKS
            if parR.Verbosity>=3
                if gpu;wait(dev);end
                tsta=tic;
            end                
            if ~isempty(amseoh);amseGamma(:,ns)=amseGamma(:,ns)+sum(vertcat(amseoh,amssoh),3);end
            if ns==Nsweeps
                xoh=bsxfun(@times,xoh,w);soh=bsxfun(@times,soh,w);poh=bsxfun(@times,poh,w);amseoh=bsxfun(@times,amseoh,w);amssoh=bsxfun(@times,amssoh,w);
                for l=1:L                
                    xh(icpr(vO(l),:),:)=xh(icpr(vO(l),:),:)+xoh(:,:,l);
                    sh(icpr(vO(l),:),:)=bsxfun(@plus,sh(icpr(vO(l),:),:),soh(:,:,l));
                    ph(icpr(vO(l),:),:)=bsxfun(@plus,ph(icpr(vO(l),:),:),poh(:,:,l));
                    amseh(icpr(vO(l),:),:)=bsxfun(@plus,amseh(icpr(vO(l),:),:),amseoh(:,:,l));
                    amssh(icpr(vO(l),:),:)=bsxfun(@plus,amssh(icpr(vO(l),:),:),amssoh(:,:,l));
                    wh(icpr(vO(l),:))=wh(icpr(vO(l),:))+w;
                end
            end
            if parR.Verbosity>=3
                if gpu;wait(dev);end
                fprintf('Time assembling blocks: %.3f s\n',toc(tsta));
            end                
        end
        if parR.Verbosity==2
            if o==1;fprintf('Processed: ');else fprintf('\b\b\b\b\b\b');end                        
            fprintf('%5.1f%%',100*min(o+bS-1,O)/O);
            if min(o+bS-1,O)==O;fprintf('\n');end
        elseif parR.Verbosity>=3
            fprintf('Processed: %5.1f%%\n',100*min(o+bS-1,O)/O);
        end
    end
    
    if ~isempty(cov) && isfield(cov,'Lambda');cov.Lambda=reshape(cov.Lambda,NC);end         
    
    if ns==Nsweeps
        %ARRANGEMENTS TO RETURN THE DATA
        wh=1./wh;
        xh=bsxfun(@times,xh,wh);sh=bsxfun(@times,sh,wh);ph=bsxfun(@times,ph,wh);amseh=bsxfun(@times,amseh,wh);amssh=bsxfun(@times,amssh,wh);
        xh=reshape(xh,NX);
        [sh,ph,amseh,amssh]=parUnaFun({sh,ph,amseh,amssh},@reshape,NX(1:dimsM));
    elseif ns==Nsweeps-1
        if parR.useRatioAMSE;amseGamma(1,:)=amseGamma(1,:)./amseGamma(2,:);end
        [valBestGamma,indBestGamma]=min(amseGamma(1,:)); 
        if parR.Verbosity>=3;fprintf('Set of AMSEs: %s\n',sprintf('%.4f ',amseGamma(1,:)));end
        if parR.Verbosity>=2;fprintf('Best AMSE: %.4f for Gamma=%.2f\n',valBestGamma,parR.Gamma(indBestGamma));end
    end
end
if parR.useRatioAMSE==1;amseh=amseh/mean(amssh(:));elseif parR.useRatioAMSE==2 amseh=cat(5,amseh,amssh);end
