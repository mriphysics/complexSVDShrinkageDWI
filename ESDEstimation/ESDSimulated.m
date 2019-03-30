function [esd,nou]=ESDSimulated(C,N,tolMinEig,NR)

%ESDSIMULATED  simulates an empirical spectral distribution using a given 
%covariance matrix
%   [ESD,NOU]=ESDSIMULATED(C,N,{TOLMINEIG},{NR})
%   * C is an MxMxOxI array containing the MxM population covariance for O 
%   distinct patches and I equally distributed populations observed
%   * N is the number of observed samples or the aspect ratio if lower or
%   equal than 1
%   * {TOLMINEIG} is the minimum allowed eigenvalue, the covariance matrix
%   is regularized by this value to guarantee positive definiteness.
%   Defaults to 0. A small value may help to stabilize the estimates
%   * {NR} is the number of realizations of the matrix (block-diagonally 
%   extended). Defaults to 1
%   * ESD is an array of size Mx1xO containing the synthesized eigenvalues
%   of the ESD in the field SIMU
%   * NOU contains the synthesized noise samples
%

if nargin<3 || isempty(tolMinEig);tolMinEig=0;end%Setting this to a low value may help for singularities at 0
if nargin<4 || isempty(NR);NR=1;end

gpu=isa(C,'gpuArray');
NC=size(C);NC(end+1:4)=1;
if N<=1;N=round(NC(1)/N);end%Second input is aspect ratio instead of number of samples
Beta=NC(1)/N;
assert(mod(N,NC(4))==0,'The number of populations observed cannot be distributed uniformly for this realization');
comp=~isreal(C);

%CONVERT TO MATRIX FORM AND GENERATE THE REQUIRED NUMBER OF SAMPLES
if NC(2)==1;isd=1;else isd=0;end
if tolMinEig~=0
    I=eye(NC(1),'like',C);
    if isd;I=diagm(I);end
    I=tolMinEig*I;
    C=bsxfun(@plus,C,I');
end
I=eye(NC(1),'like',C);

%REPEATS FOR SIMULATION
if NR>1
    if isd
        C=repmat(C,[NR 1 1]);
    else
        Cin=C;
        NCin=size(Cin);
        C=repmat(C,[NR NR 1]);
        NC=size(C);NC(end+1:4)=1;
        Cin=reshape(Cin,[NC(1:2)/NR prod(NC(3:4))]);
        C=reshape(C,[NC(1:2) prod(NC(3:4))]);    
        NCC=size(Cin,3);
        if NCC>=8
            parfor o=1:NCC;C(:,:,o)=blkdiagBody(Cin(:,:,o),NR);end
        else
            for o=1:NCC;C(:,:,o)=blkdiagBody(Cin(:,:,o),NR);end
        end    
        C=reshape(C,NC);
    end
    N=N*NR;
end

%GENERATE NOISE
n=repmat(dynInd(C,[1 1],[2 4]),[1 N 1]);
n=plugNoise(n,1);
n=resPop(n,2,[N/NC(4) NC(4)],[2 4]);
NC=size(C);NC(end+1:4)=1;
C=reshape(C,[NC(1:2) prod(NC(3:4))]);

%DIAGONALIZE
if ~isd
    U=C;D=C;

    %CAREFUL FROM HERE ON. MORE THAN ONE COVARIANCE POPULATION AND ONE REALIZATION HAS NOT BEEN TESTED
    Cin=(C+matfun(@ctranspose,C))/2;%Force the matrix to be Hermitian
    Cin=gather(Cin);
    if size(Cin,3)>=8;parforFl=Inf;else parforFl=0;end
    UId=eye(NC(1),'like',Cin);

    parfor(o=1:size(Cin,3),parforFl)
        Uor=UId;Dor=Cin(:,:,o);
        if ~isdiag(Dor)
            try
                [Uor,Dor]=schur(Cin(:,:,o));
            catch%Numerical issues-we smooth the matrix (idea from https://uk.mathworks.com/matlabcentral/answers/172633-eig-doesn-t-converge-can-you-explain-why)
                Caux=Cin(:,:,o);
                nA=sum(sum(Caux.^2));nA=nA/numel(Caux);
                Caux(Caux.^2<1e-10*nA)=0;
                Cin(:,:,o)=Caux;
                [Uor,Dor]=schur(Cin(:,:,o));           
            end            
        end
        Uaux=Uor;Daux=Dor;
        %for r=1:NR-1;Uaux=blkdiag(Uaux,Uor);Daux=blkdiag(Daux,Dor);end 
        U(:,:,o)=Uaux;D(:,:,o)=Daux;    
    end
    if gpu;U=gpuArray(U);D=gpuArray(D);end
    D=sqrt(abs(diagm(D)));

    U=bsxfun(@times,U,D);
    U=reshape(U,NC);
    n=matfun(@mtimes,U,n);
else
    n=bsxfun(@times,sqrt(abs(C)),n);
end
n=permute(n,[1 2 4 3 5]);

n=resPop(n,2:4,[N NC(3)],2:3);
nou=n;
if NC(1)<N
    n=n/sqrt((1+comp)*N);
    n=matfun(@mtimes,n,matfun(@ctranspose,n));
else
    n=n/sqrt((1+comp)*NC(1));
    n=matfun(@mtimes,matfun(@ctranspose,n),n);
end
simuESD=eigm(n);

simuESD(simuESD<0)=0;
esd.simu=double(simuESD);%Double for stability of amse computations
end

function Cn=blkdiagBody(Co,NR)   
    Cn=Co;
    for r=1:NR-1;Cn=blkdiag(Cn,Co);end   
end
