function x=spaNeigh(xx,NG)

%SPANEIGH   Builds the weights of a spatial neighborhood using a Gaussian
%N-D window
%   x=SPANEIGH(XX,NG)
%   * XX are the grid locations
%   * NG are the search dimensions of the neighborhood system
%   * X are the built weights
%

gpu=isa(xx{1},'gpuArray');

x=xx{1}(1);x=1;
for n=1:length(xx)
    perm=1:length(xx);perm(n)=1;perm(1)=n;
    r=gather(max(abs(xx{n}))); 
    %BUILD THE GAUSSIAN WINDOW
    w=single(gausswin(2*r+1));
    w=padarray(w,double(NG(n)-r),0,'both');
    if gpu;w=gpuArray(w);end
    w=permute(w,perm);
    x=bsxfun(@times,x,w);
end
