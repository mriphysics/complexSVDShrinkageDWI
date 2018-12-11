function [x,P]=ridgeDetection(x,dims,up)

%RIDGEDETECTION   Fits a linear phase to the spectrum by detecting its peak
%(slope) and the phase of the peak (intercept)
%   X=RIDGEDETECTION(X,{DIMS})
%   * X is the array on which to detect the ridge
%   * {DIMS} are the dimensions across which to detect the ridge. It defaults
%   to all dimensions in the data
%   * UP is an upsampling factor of the spectrum for more accurate location of 
%   the peak. It defaults to 1
%   * X is the ridge information
%   * P are the ridge detection parameters (constant phase and linear
%   ramps, center of coordinates is given by the first element in the
%   array)
%

if nargin<3 || isempty(up);up=1;end
ND=numDims(x);N=size(x);
if nargin<2 || isempty(dims);dims=1:ND;end
N(end+1:max(dims))=1;
ND=length(N);

gpu=isa(x,'gpuArray');if gpu;gpuF=2;else gpuF=0;end

N(dims)=N(dims)*up;
nodims=1:ND;nodims(dims)=[];
Nnodims=N;Nnodims(dims)=1;

LD=length(dims);

%A PROBLEM ONLY PARTIALLY CONSIDERED IS RINGING INTRODUCED IN THE SPECTRUM
%BY NON COMPACT SIGNALS, A BIT OF WINDOWING MAY HELP, ALSO ONE COULD USE
%THE STATISTICS OF THE PHASE TO MODULATE BY A SPATIAL PROFILE DIFFERENT
%FROM THE SIGNAL MAGNITUDE, ALTHOUGH THAT COULD BE DONE OUTSIDE THIS
%FUNCTION
x=resampling(x,N,2);
for n=1:LD;x=fftGPU(x,dims(n),gpuF);end

x=resSub(x,dims);
[x,ix]=max(x,[],dims(1));
x=reshape(x,Nnodims);
ix=ind2subV(N(dims),ix);

%PARAMETERS
P=real(x);P(:)=0;
repm=ones(1,max(numDims(P),dims(1)));repm(dims(1))=LD+1;
P=repmat(P,repm);
for n=1:LD;P=dynInd(P,1+n,dims(1),resPop(wrapToPi(2*pi*(dynInd(ix,n,2)-1)/N(dims(n))),1,Nnodims(nodims),nodims));end

F=buildStandardDFTM(N(dims),0,gpu);
for n=1:LD
    F{n}=F{n}(ix(:,n),:);
    F{n}=shiftdim(F{n},-ND);
    F{n}=resPop(F{n},ND+1,Nnodims(nodims),nodims);
    F{n}=resPop(F{n},ND+2,N(dims(n)),dims(n));
    
    NF=size(F{n});
    NF(dims(n))=NF(dims(n))/up;
    F{n}=resampling(F{n},NF,2);
end
x=sign(x);
for n=1:LD;x=bsxfun(@times,x,conj(F{n}));end
P=dynInd(P,1,dims(1),dynInd(angle(x),ones(1,LD),dims));
