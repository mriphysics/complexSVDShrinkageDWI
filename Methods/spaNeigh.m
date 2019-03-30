function x=spaNeigh(xx,NG,typ)

%SPANEIGH   Builds the weights of a spatial neighborhood using a Gaussian
%or a uniform N-D window
%   x=SPANEIGH(XX,NG,{TYP})
%   * XX are the grid locations
%   * NG are the search dimensions of the neighborhood system
%   * {TYP} is the type of window: Gauss for Gaussian (default) / Unifo for
%   Uniform
%   * X are the built weights
%

if nargin<3;typ='Gauss';end

gpu=isa(xx{1},'gpuArray');

x=xx{1}(1);x=1;
for n=1:length(xx)
    perm=1:length(xx);perm(n)=1;perm(1)=n;
    r=gather(max(abs(xx{n}))); 
    %BUILD THE GAUSSIAN WINDOW    
    w=single(gausswin(2*r+1));
    w=padarray(w,double(NG(n)-r),0,'both');
    if gpu;w=gpuArray(w);end
    if strcmp(typ,'Unifo');w(:)=1;end
    w=permute(w,perm);
    x=bsxfun(@times,x,w);
end
