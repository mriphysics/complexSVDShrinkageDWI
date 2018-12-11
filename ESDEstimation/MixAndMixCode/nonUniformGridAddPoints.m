function xN=nonUniformGridAddPoints(x,y,NP,typGrid,parGrid)

%NONUNIFORMGRIDADDPOINTS adds a given number of points to an existing grid according
%to a specific point distribution criterion
%   [X,Y]=NONUNIFORMGRIDADDPOINTS(X,Y,NP,TYPGRID,PARGRID)
%   * X is an existing set of grids of points with grid points arranged in
%   the rows and grids arranged in other dimensions
%   * Y are the values of a function over X
%   * NP is the number of points to be added to the grid
%   * TYPGRID is the type of grid to be constructed. Allowed generic 
%   types are UNIFORM for uniform grids; CAMPYLOTROPIC for 
%   campylotropically weighted grids according to [1] CM Ablow and S 
%   Schechter, "Campylotropic coordinates," J Comput Phys, 27:351-362, 
%   1978; ARCLENGTH for arclength weighted grids as in [2] AB White, "On
%   selection of equidistributing meshes for two-point boundary-value
%   problems," SIAM J Num Anal, 16(3):472-502, 1979, where other 
%   possibilities were considered, and more generally see for instance [3]
%   JF Thompson, "A survey of dinamically-adaptive grids in the numerical 
%   solution of partial differential equations," Apl Num Math, 1:3-27,
%   1985. Specific weight functions are to be implemented for ESD
%   estimation. Currently we have implemented the SECONDEXPECT weight, 
%   which subdivides on the basis of the square root of the product of the 
%   grid location and the second order derivative, which is expected to 
%   improve approximation to the edges of the EDS (for non very long 
%   tailed distributions) and EXPECT which subdivides on the basis of the 
%   grid location, which is expected to improve shrinkage accuracy.
%   * PARGRID is a grid generation parameter (currently only for 
%   campylotropic coordinates (0 if empty, so they reduce to arclength)
%   * XN are the added grid points
%

if ~exist('parGrid','var') || isempty(parGrid);parGrid=0;end
gpu=isa(x,'gpuArray');

NPG=size(x);
resV=[NPG(1) prod(NPG(2:end))];
[x,y]=parUnaFun({x,y},@reshape,resV);
NPGR=size(x);
%FIRST ORDER DERIVATIVES
xd=diff(x,1,1);
yd=diff(y,1,1)./xd;
xdw=xd;
xd=x(1:end-1,:)+xd/2;
if strcmp(typGrid,'Campylotropic') || strcmp(typGrid,'SecondExpect')
    %SECOND ORDER DERIVATIVES
    xdd=diff(xd,1,1);
    ydd=diff(yd,1,1)./xdd;
    xdd=xd(1:end-1,:)+xdd/2;
    ydd=padarray(ydd,[1 0],0);
    xdd=vertcat(x(1,:),xdd,x(end,:));
    %INTERPOLATION OVER THE FIRST DERIVATIVE GRID
    yddi=interp1GPU(xdd,ydd,xd);
end
%INTERPOLATION OVER THE FIRST DERIVATIVE GRID
%yi=yd;
%for o=1:NPGR(2);yi(:,o)=interp1(x(:,o),y(:,o),xd(:,o),'linear',0);end
%WEIGHT DEFINITION
w=yd;
if strcmp(typGrid,'Uniform')
    w(:)=1;
elseif strcmp(typGrid,'ArcLength')
    w=sqrt(1+yd.^2);
elseif strcmp(typGrid,'Campylotropic')
    w=sqrt(1+yd.^2);
    w=w.*(1+parGrid*abs(yddi)./w.^3);
elseif strcmp(typGrid,'SecondExpect')
    w=sqrt((abs(yddi).*xd));
    %w=abs(yddi);
    %w=(abs(yddi).^parGrid).*(xd.^(1-parGrid));
elseif strcmp(typGrid,'Expect')
    w=xd;
end
w=w.*xdw;
%w=(abs(ydd).^Beta).*(x.^(1-Beta));
w=gather(w);

%SUBDIVISION OF GRID CELLS
subx=gridSubdivide(w,NP);

[x,subx]=parUnaFun({x,subx},@gather);

%CREATION OF NEW GRID POINTS
xN=fillGridPoints(x,subx,NP);

if gpu;xN=gpuArray(xN);end
xN=reshape(xN,[NP NPG(2:end)]);
