function indS=arraySupport(x,tol,ext,typ)

%ARRAYSUPPORT   Finds the minimum and maximum indexes determining the
%support of the first dimension of an array
%   INDS=ARRAYSUPPORT(X,{TOL},{EXT},{TYP})
%   * X is the array
%   * {TOL} is a tolerance for support definition
%   * {EXT} extends the support by a given amount
%   * {TYP} is the type of support to be returned. 1 (default) for global
%   indexes; 0 for indexes referred to the first dimension
%   % INDS are the indexes of support
%

if ~exist('tol','var') || isempty(tol);tol=0;end
if ~exist('ext','var') || isempty(ext);ext=0;end
if ~exist('typ','var') || isempty(typ);typ=1;end

x=(abs(x)>tol);
[~,thI]=max(x,[],1);
[~,thS]=max(flip(x,1),[],1);
N=size(x);
thS=N(1)-thS+1;
[thI,thS]=parUnaFun({thI,thS},@double);
thII=max(thI-ext,1);
thSS=min(thS+ext,N(1));
indS=vertcat(thII,thSS);
if typ==1
    NG=prod(N(2:end));
    vNG=(0:NG-1)';
    vNG=reshape(vNG,[1 N(2:end)]);
    indS=bsxfun(@plus,indS,vNG*N(1));
end
