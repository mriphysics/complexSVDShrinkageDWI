function esd=ESDMixAndMix(C,N,tolEnd,NL,tolMinEig,indbldiag,tolSta,factCool,M,tolUnique,nMax)

%ESDMIXANDMIX  computes the ESD using fixed point iterations on a given 
%covariance matrix. It generalizes the Spectrode code in [1] E Dobriban, 
%"Efficient computation of limit spectra of sample covariance matrices," 
%Random Matrices: Theory Appl, 4(4):1550019-36 p, 2015 in that it allows to
%estimate ESDs for distinct per observation (or per group of observations) 
%population covariances, a "MIXture of populations" model, using the 
%fixed-point equation proposed in [2] S Wagner et al, "Large System 
%Analysis of Linear Precoding in Correlated MISO Broadcast Channels Under 
%Limited Feedback," IEEE TIT, 58(7):4509-4537, 2012. The method has the 
%following additional main differences with the Spectrode approach: (1) 
%GPU computation is used; (2) homotopy continuation is used to approximate 
%the Stieltjes transform of the ESD law over the real line by iteratively 
%decreasing the imaginary component where the Stieltjes transform is 
%approximated; (3) ANDerson MIXing following [3] DG Anderson, "Iterative 
%procedures for nonlinear integral equations," J ACM, 12:537-560, 1965 is 
%used to accelerate convergence of the fixed point equations; [4] adaptive 
%gridding is used for better approximation at the egdes of the ESD.
%   ESD=ESDMIXANDMIX(C,N,{TOLEND},{TOLMINEIG},{PERCSAM}{INDBLDIAG},{TOLSTA},{FACTCOOL},{M},{TOLUNIQUE})
%   * C is an MxMxOxI array containing the MxM population covariance 
%   (alternatively a Mx1 diagonal population covariance) for O distinct
%   items of the problem (patches in the context of DWI signal prediction)
%   for the I distributed populations ocurring with equal chances in the
%   observed data
%   * N is the number of observed samples or the aspect ratio if lower or
%   equal than 1
%   * {TOLEND} is the error tolerance, defaults to 1e-4
%   * {NL} is the number of levels of the adaptive grid subdivision.
%   Defaults to 1
%   * {TOLMINEIG} is the minimum allowed eigenvalue, the covariance matrix
%   is regularized by this value to guarantee positive definiteness.
%   Defaults to 1e-3. A small value generally helps to stabilize the estimates
%   * {INDBLDIAG} indicates the independence structure of the matrices
%   (sizes of independent block diagonal submatrices) to accelerate
%   computations. Defaults to empty
%   * {TOLSTA} is the starting error tolerance where to start homotopy in
%   the imaginary direction. Defaults to 1
%   * {FACTCOOL} is the factor for homotopy cooling in the complex plane.
%   Defaults to 2
%   * {M} is the number of iterations for Anderson mixing (0 does not use 
%   Andersson mixing). Defaults to 3
%   * {TOLUNIQUE} is the tolerance to group population eigenvalues, 
%   defaults to 0
%   * {NMAX} is the maximum number of iterations of fixed point. It
%   defaults to 100
%   * ESD is a cell array with the estimated ESDs. Each element is a 
%   structure containing the following fields:
%       - ESD.GRID, the grid on which the empirical spectral distribution is
%   defined
%       - ESD.DENS, the empirical spectral distribution density
%       - ESD.THRE, the upper bound on the empirical spectral distribution
%       - ESD.GRIDD, the gradient of the grid locations
%       - ESD.APDF, the accumulated estimated pdf (for consistent moment
%       computation later on).
%

if nargin<3 || isempty(tolEnd);tolEnd=1e-4;end
if nargin<4 || isempty(NL);NL=1;end
if nargin<5 || isempty(tolMinEig);tolMinEig=1e-3;end
if nargin<6;indbldiag=[];end
if nargin<7 || isempty(tolSta);tolSta=1;end
if nargin<8 || isempty(factCool);factCool=10;end%10;end%It seems it may be feasible to run it with a factCool of 10, but more tests are required. We are being really aggressive due to the nature of our application, reduce this to 2 for more reliable estimates
if nargin<9 || isempty(M);M=2;end%This may probably be something along min(C(4)/2,Mmax)
if nargin<10 || isempty(tolUnique);tolUnique=0;end%1e-2;end
if nargin<11 || isempty(nMax);nMax=100;end

%[C,N,tolEnd,tolSta]=parUnaFun({C,N,tolEnd,tolSta},@double);

gpu=isa(C,'gpuArray');

%HARDCODED PARAMETERS
NS=ones(1,NL);%Number of points to be added at each subdivision of the adaptive multirresolution scheme as a ratio over 3M (number of points at first level)
typGrid='SecondExpect';%Criterion to build weights for adaptive grids
Nin=15;%Minimum number of points to subdivide the search within each detected interval
Nfu=3;%Minimum number of points per eigenvalue on the grid

%CHECKS
NC=size(C);NC(end+1:4)=1;
if N<=1;N=round(NC(1)/N);end%Second input is aspect ratio instead of number of samples
Beta=NC(1)/N;
assert(mod(N,NC(4))==0,'The number of populations observed cannot be distributed uniformly for this realization');
NL=NL+1;
assert(NL<=7,'More than 7 subdivision levels (%d requested) are numerically unstable',NL);
%assert(NL>=2,'A minimum of 2 subdivision levels is required while %d were requested',NL);

%DIAGONALIZE WHILE ASSURING IT IS POSITIVE DEFINITE
if NC(2)==1;isd=1;else isd=0;end
I=eye(NC(1),'like',C);
if isd;I=diagm(I);end
I=tolMinEig*I;
C=bsxfun(@plus,C,I');
I=eye(NC(1),'like',C);
if ~isd
    eigv=eigm(C);    
else
    eigv=C;
    if NC(4)~=1;C=diagm(permute(C,[2 1 3 4]));end%Rather arbitrary assumption
end
eigv=abs(eigv);

%GROUP EIGENVALUES
grEig=0;
if prod(NC(3:4))==1%Single spectrum computation    
    weigv=eigv;weigv(:)=1/NC(1);
    [eigvtol,~,itol]=uniquetol(gather(eigv),gather(tolUnique*max(eigv)));
    if gpu;eigvtol=gpuArray(eigvtol);end
    if length(eigvtol)<NC(1)
        weigvtol=accumarray(itol,weigv);
        weigeigvtol=eigvtol.*weigvtol;
        grEig=1;
    end
end


%WE COMPUTE THE INITIAL GRID
N0=size(eigv,1);
if NC(4)==1
    eigG=eigv;
else
    eigG=permute(eigv,[1 4 3 2]);
    NE=size(eigG);NE(end+1:3)=1;
    eigG=reshape(eigG,[prod(NE(1:2)) 1 NE(3)]);
    eigG=cat(1,eigG,eigm(mean(C,4)));
end
gridx=startingGrid(eigG,Beta,N0,Nin,Nfu);
perm=1:5;perm(1)=5;perm(5)=1;%Generic permutation
gridx=permute(gridx,perm);

%AUXILIARY VARIABLES
IdM=eye(NC(1:2),'like',C);%For the resolvent with more than one covariance along the columns
repPop=ones(1,5);repPop(4)=NC(4);%Auxiliary variable
%NP=ceil(NC(1)*NS(1));
NP=size(gridx,5);
gridxPre=[];vPre=[];densPre=[];
tolSta=single(tolSta);
if gpu;tolSta=gpuArray(tolSta);end

%BLOCK DIAGONAL ACCELERATION
if ~isempty(indbldiag) && NC(4)~=1
    ce=length(indbldiag);
    C=mat2bldiag(C,indbldiag);
    IdM=mat2bldiag(IdM,indbldiag);
else
    ce=0;
end

%MULTIRRESOLUTION SOLVER
for l=1:NL%For each resolution level  
    
    %GRIDS AT THIS RESOLUTION LEVEL
    if l>1
        [gridxPre,densPre]=parUnaFun({gridxPre,densPre},@permute,perm);
        gridx=nonUniformGridAddPoints(gridxPre,densPre,round(NP*NS(l-1)),typGrid);
        [gridxPre,densPre,gridx]=parUnaFun({gridxPre,densPre,gridx},@permute,perm); 
    end
    
    permG=1:5;permG(numDims(gridx))=5;permG(5)=numDims(gridx);
    gridx=permute(gridx,permG);    
    grid=bsxfun(@plus,gridx,1i*tolSta(1).^2);
    
    %INITIALIZE / INTERPOLATE ARRAYS
    if l==1
        v=-1./(Beta*grid);
        v=repmat(v,repPop);
    else%if l==2%Linear interpolation  
        [gridPreU,gridCur,v]=parUnaFun({gridxPre,gridx,vPre},@permute,perm);
        [gridPreU,gridCur]=parUnaFun({gridPreU,gridCur},@repmat,repPop);                
                
        NG=size(gridPreU);
        NGG=prod(NG(2:end));        
        [gridPreU,v]=parUnaFun({gridPreU,v},@reshape,[NG(1) NGG]);        
        gridCur=reshape(gridCur,[NP NGG]);
        vO=interp1GPU(gridPreU,v,gridCur);
        v=reshape(vO,[NP NG(2:end)]);v=permute(v,perm);
    end         

    ffgg=repmat(grid,[2 M+1 1 NC(4)]);
    NGG=size(grid,5);
    
    [ffgg,v]=parUnaFun({ffgg,v},@permute,[1 2 3 5 4]);
    ffgg=reshape(ffgg,[2 M+1 NC(3)*NGG NC(4)]);
    
    grid=reshape(grid,[1 1 NC(3)*NGG]);
    v=reshape(v,[1 1 NC(3)*NGG NC(4)]);
    tolSta=repmat(tolSta(1),[1 1 NC(3)*NGG]);
    tolUse=tolSta;
    convItem=logical(gather(tolSta));convItem(:)=false;%Flag to compute
    errpre=tolSta;errpre(:)=inf;
    
    if NC(4)==1 && grEig;[eigvtolC,weigvtolC,weigeigvtolC]=parUnaFun({eigvtol,weigvtol,weigeigvtol},@repmat,[1 1 NGG]);
    elseif NC(4)==1;eigvC=repmat(eigv,[1 1 NGG]);
    else
        if ce
            CC=C;
            for c=1:ce
                CC{c}=repmat(CC{c},[1 1 1 1 NGG]);
                CC{c}=permute(CC{c},[1 2 3 5 4]);
                NNC=size(CC{c});
                CC{c}=reshape(CC{c},[NNC(1:2) NC(3)*NGG NC(4)]);
            end
            Ci=cell(1,ce);
        else
            CC=repmat(C,[1 1 1 1 NGG]);
            CC=permute(CC,[1 2 3 5 4]);
            CC=reshape(CC,[NC(1:2) NC(3)*NGG NC(4)]);
        end
    end

    
    n=0;
    vn=v;
    while n<nMax
        convItemF=find(~convItem);
        ffggi=ffgg(:,:,convItemF,:);vi=v(:,:,convItemF,:);gridi=grid(:,:,convItemF);
        %FIXED POINT EQUATIONS        
        if NC(4)==1
            if exist('weigvtol','var')
                vni=sum(bsxfun(@rdivide,weigeigvtolC(:,:,convItemF),bsxfun(@minus,bsxfun(@rdivide,eigvtolC(:,:,convItemF),1+Beta*vi),gridi)),1);%There are some numerical differences with the next case due to the application of the mean (specially for single data)
            else
                eigvi=eigvC(:,:,convItemF);
                vni=mean(bsxfun(@rdivide,eigvi,bsxfun(@minus,bsxfun(@rdivide,eigvi,1+Beta*vi),gridi)),1);
            end
        else
            if ~ce
                Ci=CC(:,:,convItemF,:);
                Ri=mean(bsxfun(@rdivide,Ci,1+vi*Beta),4)-bsxfun(@times,IdM,gridi);                         
                vni=multDimSum(bsxfun(@times,matfun(@transpose,Ci),matfun(@inv,Ri)),1:2)/NC(1);
            else
                vni=vi;vni(:)=0;
                for c=1:ce
                    Ci{c}=CC{c}(:,:,convItemF,:);
                    Ri=mean(bsxfun(@rdivide,Ci{c},1+vi*Beta),4)-bsxfun(@times,IdM{c},gridi);                             
                    vni=vni+multDimSum(bsxfun(@times,matfun(@transpose,Ci{c}),matfun(@inv,Ri)),1:2);
                end
                vni=vni/NC(1);
            end
        end
        %ANDERSON MIXING
        ffggi(2,M+1,:,:)=vni;
        Mi=min(M,n);
        [ffggi,vni]=andersonMixing(ffggi,vi,Mi);        
        ffgg(:,:,convItemF,:)=ffggi;v(:,:,convItemF,:)=vi;vn(:,:,convItemF,:)=vni;   
        %CONVERGENCE 
        vnic=imag(vn)<0;vnic=find(vnic);
        vn(vnic)=conj(vn(vnic));
        
        errcur=multDimMax(abs(vn-v),[1:2 4]);
        v=vn;
        erric=errcur<=errpre;erric=find(erric);
        %erric=errcur<=tolUse;erric=find(erric);
        tolUse(erric)=max(tolUse(erric)/factCool,tolEnd);
        grid=bsxfun(@plus,real(grid),1i*tolUse.^2);
        ercte=errcur(:)<tolEnd;
        erto=abs(tolUse(:)-tolEnd)<tolEnd/10;
        if n>0;convItem(ercte & erto)=1;end
        if n>0 && all(ercte) && all(erto);break;end
        errpre=errcur;
        n=n+1;
        %if n==nMax;fprintf('Not converged:%s\n',sprintf(' %d',find(~convItem)));end
    end
    
    %DENSITY ESTIMATE
    if NC(4)==1 && exist('weigvtol','var')
        m=sum(bsxfun(@rdivide,weigvtolC,bsxfun(@minus,bsxfun(@rdivide,eigvtolC,1+Beta*v),grid)),1);
    elseif NC(4)==1
        m=mean(1./bsxfun(@minus,bsxfun(@rdivide,eigvC,1+Beta*v),grid),1);
    else
        if ~ce
            m=mean(diagm(matfun(@inv,mean(bsxfun(@rdivide,CC,1+v*Beta),4)-bsxfun(@times,IdM,grid))),2);
        else
            m=zeros([1 1 NC(3)*NGG],'like',v);
            for c=1:ce;m=m+sum(diagm(matfun(@inv,mean(bsxfun(@rdivide,CC{c},1+v*Beta),4)-bsxfun(@times,IdM{c},grid))),2);end
            m=m/NC(1);
        end            
    end
    m=reshape(m,[1 1 NC(3) 1 NGG]);
    v=reshape(v,[1 1 NC(3) NGG NC(4)]);
    v=permute(v,[1 2 3 5 4]);            
    dens=imag(m)/pi;
    
    %ADDING NEW SAMPLES AND SORTING
    if l==1
        gridxPre=gridx;
        vPre=v;
        densPre=dens;
    else
        gridxPre=cat(5,gridxPre,gridx);densPre=cat(5,densPre,dens);vPre=cat(5,vPre,v);
        [gridxPre,iSGr]=sort(gridxPre,5);
        [densPre,vPre]=parUnaFun({densPre,vPre},@indDim,iSGr,5);
    end    
    %fprintf('Number of iterations in level %d: %d\n',l,n);
    %fprintf('Tolerance achieved in level %d: %.1e\n',l,max(tolUse(:)));
end

%ASSIGNMENT
[esd.grid,esd.dens]=parUnaFun({gridxPre,abs(densPre)},@permute,[5 2 3 4 1]);
[esd.grid,esd.dens]=parUnaFun({esd.grid,esd.dens},@single);

%UPPER THRESHOLD ESTIMATE
NP=size(esd.dens,1);
thS=dynInd(arraySupport(esd.dens,tolEnd(end),0,0),2,1);
thSS=min(thS+1,NP);
esd.thre=(indDim(esd.grid,thS,1)+indDim(esd.grid,thSS,1))/2;
esd.dens(esd.grid>repmat(esd.thre,[NP 1 1]))=0;

%GRADIENT AND INTEGRAL ESTIMATES
esd.gridd=permute(gradient(permute(esd.grid,[2 1 3])),[2 1 3]);
esd.apdf=sum(esd.dens.*esd.gridd,1);

