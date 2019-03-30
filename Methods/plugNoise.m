function x=plugNoise(x,useR)

% PLUGNOISE plugs complex circularly symmetric AWGN noise samples in an 
%array
%   X=PLUGNOISE(X,USER) 
%   * X is the input array
%   * USER enables the usage of real only noise
%   * X is the noise array with same dimensions as the input array
%

if nargin<2 || isempty(useR);useR=0;end%Complex only, for back-compatibility!

comp=~isreal(x) || ~useR;
N=size(x);
if comp;x=randn(N,'like',real(x))+1i*randn(N,'like',real(x));else x=randn(N,'like',x);end

