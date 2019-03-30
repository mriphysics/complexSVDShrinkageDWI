%COMPLEXSVDSHRINKAGEDWI_EXP4 script performs the experiment included in 
%Fig. 5 of the manuscript ''Complex diffusion-weighted image estimation 
%via matrix recovery under general noise models'', L. Cordero-Grande, D. 
%Christiaens, J. Hutter, A.N. Price, and J.V. Hajnal

clearvars
parR.Plot=0;%0 to save results / 1 to save and plot results
parR.Verbosity=2;%Level of verbosity, from 0 to 3
gpu=gpuDeviceCount;%Detects whether gpu computations are possible
if gpu;dev=gpuDevice;end

curFolder=fileparts(mfilename('fullpath'));
addpath(genpath(curFolder));%Add code
pathData=strcat(curFolder,'/../complexSVDShrinkageDWIData');

%READ DATA
if parR.Verbosity>0
    fprintf('Experiment on simulated data\n');
    fprintf('Reading input data...\n');
end
nii=load_untouch_nii(sprintf('%s/dwix.nii',pathData));%LOAD GROUND-TRUTH
x=single(nii.img);
if gpu;x=gpuArray(x);end
N=size(x);N(end+1:4)=1;
nii=load_untouch_nii(sprintf('%s/dwim.nii',pathData));%LOAD MASK
M=single(nii.img>0.5);
if gpu;M=gpuArray(M);end
nii=load_untouch_nii(sprintf('%s/dwiw.nii',pathData));%LOAD THE NOISE PROFILES
UpsilonInv=nii.img;nii=[];
if gpu;UpsilonInv=gpuArray(UpsilonInv);end
nii_or=load_untouch_nii(sprintf('%s/dwiy.nii',pathData));%LOAD NOISY IMAGE
y=single(nii_or.img);
if gpu;y=gpuArray(y);end 
if parR.Verbosity>0;fprintf('Finished reading input data\n');end

%RUN NLAM METHOD
if parR.Verbosity>0
    if gpu;wait(dev);end
    tsta=tic;
end
%system(sprintf('%s/nlsamRun.sh %s',pathData,pathData));
if parR.Verbosity>0
    if gpu;wait(dev);end
    tend=toc(tsta);
    fprintf('Retrieval time using NLSAM method: %.2fs\n',tend);
end

%EXPERIMENT PARAMETERS
typExec='Quick';%For quick computations to inspect behaviour and results, any other string, for instance 'Reproduce' will strictly reproduce the experiment in the paper
NoiEstMethV={'None','Exp2','None'};%See eqs. 21 and 22, one of the following: 'None' / 'Exp1' / 'Exp2' / 'Medi'
ShrinkMethV={'Frob','Exp2','Frob'};%Frob stands for Frobenius cost. One of the following: 'Hard' / 'Soft' / 'Frob' / 'Oper' / 'Nucl' / 'Exp1' / 'Exp2'
WeightAssembV={'Invva','Unifo','Unifo'};%Type of window weighting for patch assembling. One of the following: 'Gauss' / 'Unifo' / 'Invva'

%MAIN PARAMETERS
cov.dimPE=0;
cov.dimNoPE=[0;0];
voxsiz=nii_or.hdr.dime.pixdim(2:4);
parR.UseComplexData=false;%false for not to use complex data
parR.useRatioAMSE=2;%To compute the NAMSE (1), AMSE (0) or both AMSE and AMSS (square of signal level) (2)
parR.ESDMeth=3;%0->Use simulation / 1-> Use SPECTRODE / 2-> Use MIXANDMIX / 3-> Force computation of simulated spectra (to get AMSE estimates when using noise estimates instead of noise modeling)
parR.ESDTol=1e-2;%Tolerance for ESD computations (role depends on the method)
parR.NR=1;%Number of realizations of random matrix simulations
parR.DirInd=0;%To accelerate by using independence of covariances along a given direction
parR.PatchSimilar=2;%Similarity metric to build the patches. 2 for Euclidean / 1 for Manhattan
parR.Gamma=0.2:0.05:0.95;%Random matrix aspect ratio, vector of values interpreted as candidates for patch size estimations
%parR.Gamma=cat(2,parR.Gamma,flip(1./parR.Gamma));%To test the whole range of aspect ratios
parR.Subsampling=[2 2 2];%Subsampling factor for patch construction, bigger values produce quicker denoising, but if too big holes may appear in the resulting data
if strcmp(typExec,'Quick');parR.Subsampling=[4 4 4];end
parR.DrawFact=3;%Factor to multiply the subsampling to sweep the image space to estimate patch sizes

%NUMERIC PARAMETERS
thNoise=1e-3;%Mask out below this value

strMet={'GSVS','MPPCA','GSVS-stdhat'};
NN=length(NoiEstMethV);%Noise estimation method    
for n=1:NN%First ours, second Veraart, third ours with noise estimation from Veraart
    if parR.Verbosity>0
        fprintf('Retrieval using %s method...\n',strMet{n});
        if gpu;wait(dev);end
        tsta=tic;
    end              
    parR.NoiEstMeth=NoiEstMethV{n};parR.ShrinkMeth=ShrinkMethV{n};parR.ShrinkMeth=ShrinkMethV{n};
    parR.WeightAssemb=WeightAssembV{n};

    yi=y;cov.Lambda=UpsilonInv.^2;
    %ADD MEAN LEVEL OF NOISE OUTSIDE THE MASK SO THAT NOISE ESTIMATION BECOMES INDEPENDENT FROM MASKING
    mUps=mean(UpsilonInv(M==1));%Mean noise
    if ~strcmp(parR.NoiEstMeth,'None')            
        noi=mUps*plugNoise(x,1);
        yi=bsxfun(@times,yi,M)+bsxfun(@times,noi,1-M);noi=[];
        cov.Lambda=bsxfun(@times,cov.Lambda,M)+bsxfun(@times,mUps.^2,1-M);noi=[];
    end

    %SVD PATCH BASED RECOVERY
    [xhat,stdhat,~,amsehat,gammahat]=patchSVShrinkage(yi,3,voxsiz,parR,cov,[],M);

    %MASK
    if ~strcmp(parR.NoiEstMeth,'None');xhat=bsxfun(@times,xhat,M);end        

    if parR.Verbosity>0
        if gpu;wait(dev);end
        fprintf('Retrieval time using %s method: %.2fs\n',strMet{n},toc(tsta));            
    end  
    parR.Gamma=gammahat;%To use the same aspect ratio for each compared case
    if strcmp(parR.NoiEstMeth,'Exp2');UpsilonInv=bsxfun(@times,stdhat,M);end%To use the estimated noise levels for denoising

    %SAVE DATA
    if parR.Verbosity>0
        if gpu;wait(dev);end
        tsta=tic;
    end 
    nii.hdr=nii_or.hdr;nii.img=gather(xhat);%RETRIEVED IMAGE
    nii.hdr.dime.dim=[4 N ones(1,3)];   
    save_nii(nii,sprintf('%s/dwix%s.nii',pathData,strMet{n}));
    if parR.Verbosity>0
        if gpu;wait(dev);end       
        fprintf('Time saving %s: %.2fs\n',strMet{n},toc(tsta));
    end
    %nii.hdr=nii_or.hdr;nii.img=gather(permute(amsehat,[1 2 3 5 4]));%ASYMPTOTIC MEAN SQUARE ERROR
    %nii.hdr.dime.dim=[4 N(1:3) 2 ones(1,3)];
    %save_nii(nii,sprintf('%s/dwiams%s.nii',pathData,strMet{n}));            
end

%PLOT RESULTS
if parR.Plot;plot_Exp4;end

