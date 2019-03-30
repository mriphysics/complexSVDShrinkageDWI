function esd=ESDSpectrode(C,N,tolEnd,tolMinEig,tolUnique)

%ESDSPECTRODE  wrapper to call the Spectrode method described in [1] E 
%Dobriban, "Efficient computation of limit spectra of sample covariance 
%matrices," Random Matrices: Theory Appl, 4(4):1550019-36 p, 2015, based on
%the implementation at https://github.com/dobriban/eigenedge to compute a 
%set of O empirical sample distributions (ESD) from O population 
%covariances (corresponding to the patches in the context of DWI signal 
%prediction) arranged along the third dimension with corresponding atomic 
%distributions with M equiprobable components arranged in the rows or MxM 
%covariance matrices from which to compute them 
%   ESD=ESDSPECTRODE(C,N,{TOLEND},{TOLMINEIG},{TOLUNIQUE})
%   * C is the population covariance (alt atomic distribution column array)
%   * N is the number of observed samples or the aspect ratio if lower or
%   equal than 1
%   * {TOLEND} is the error tolerance, defaults to 1e-4
%   * {TOLMINEIG} is the minimum allowed eigenvalue, the covariance matrix
%   is regularized by this value to guarantee positive definiteness.
%   Defaults to 0. A small value may help to stabilize the estimates
%   * {TOLUNIQUE} is the tolerance to group population eigenvalues, defaults 
%   to 0
%   * ESD is a cell array with the estimated ESDs. Each element is a 
%structure containing the following fields:
%       - ESD.GRID, the grid on which the empirical spectral distribution is
%   defined
%       - ESD.DENS, the empirical spectral distribution density
%       - ESD.THRE, the upper bound on the empirical spectral distribution
%       - ESD.GRIDD, the gradient of the grid locations
%       - ESD.APDF, the accumulated estimated pdf (for consistent moment
%       computation later on).
%

if nargin<3 || isempty(tolEnd);tolEnd=1e-4;end
if nargin<4 || isempty(tolMinEig);tolMinEig=0;end%Setting this to a low value may help for singularities at 0
if nargin<5 || isempty(tolUnique);tolUnique=0;end

%DIAGONALIZE
NC=size(C);NC(end+1:3)=1;
assert(length(NC)<=3,'Covariance defined over %d dimensions, but only 3 are accepted',length(NC));

%DIAGONALIZE WHILE ASSURING IT IS POSITIVE DEFINITE
if NC(2)==1;isd=1;else isd=0;end
I=eye(NC(1),'like',C);
if isd;I=diagm(I);end
I=tolMinEig*I;
C=bsxfun(@plus,C,I');
I=eye(NC(1),'like',C);
if ~isd;eigv=eigm(C);else eigv=C;end

if N<=1;N=round(NC(1)/N);end%Second input is aspect ratio instead of number of samples
Beta=NC(1)/N;
eigv=gather(double(abs(eigv)));

esd=cell(1,NC(3));
w=ones(NC(1),1)/NC(1);

if NC(3)>=8
    parfor o=1:NC(3);esd{o}=SpectrodeBody(eigv(:,1,o),tolUnique,w,Beta,tolEnd);end
else
    for o=1:NC(3);esd{o}=SpectrodeBody(eigv(:,1,o),tolUnique,w,Beta,tolEnd);end
end
        
end

function esdo=SpectrodeBody(eigvo,tolUnique,w,Beta,tolEnd)
    %GROUP EIGENVALUES
    [eigvtol,~,itol]=uniquetol(eigvo,tolUnique*max(eigvo));
    wtol=accumarray(itol,w);
    %CALL THE SPECTRODE METHOD
    if any(eigvtol(:)>1e-6)
        [grid,dens,~,~,mass_at_0,K_hat,l_hat,u_hat]=spectrode(eigvtol,Beta,wtol,[],[],tolEnd);  
        esdo.grid=double(grid);esdo.dens=double(dens);esdo.thre=double(max(u_hat));%Doubles for stability of amse computations
        esdo.gridd=gradient(esdo.grid);
        esdo.apdf=sum(esdo.dens.*esdo.gridd);
    else
        esdo=[];
    end  
end
