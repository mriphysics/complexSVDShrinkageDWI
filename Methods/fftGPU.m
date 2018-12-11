function x=fftGPU(x,m,gpu,F,re)

%FFTGPU   Configurable GPU-based FFT computation
%   X=FFTGPU(X,M,GPU,{F})
%   * X is the array on which to apply the FFT
%   * M is the direction along which to apply the FFT
%   * GPU is a flag that determines whether to use cpu/gpu with built-in
%   matlab functions (0) / kernel gpu (1), kernel matrix-based gpu (2), gpu
%   (3) or matrix-based gpu computation (4). Some modifications are 
%   introduced to the user preferences due to bugs in gpu computations in 
%   matlab 2015a (roughly speaking these affected the gpu computation of 
%   the fft for certain ---usually prime numbers--- sizes of the array when 
%   the computation was not performed along the first dimension), so users
%   should better test the matlab gpu fft behaviour in their systems with 
%   with different array sizes and along different dimensions
%   * {F} is a FFT matrix (or any other square matrix) provided by the 
%   user. If not provided and required, it is obtained by the function
%   * {RE} is a flag to indicate that the FFT is to be applied to a real 
%   image (defaults to 0)
%   * X is the FFT-transformed array
%

if strcmp(version('-release'),'2015a') && gpu==0;gpu=1;end

if nargin<4;F=[];end
if nargin<5 || isempty(re);re=0;end

ism=1;
if gpu==0 && ~re
    x=fft(x,[],m);
elseif gpu==1 && ~re
    if m==1
        if size(x,1)~=1;x=fft(x);end
    else
        perm=1:ndims(x);perm(1)=m;perm(m)=1;
        x=permute(x,perm);
        if size(x,1)~=1;x=fft(x);end
        x=permute(x,perm);
    end
elseif gpu==2 || re
    N=size(x,m);
    if N~=1 || re>0
        if isempty(F);F=build1DFTM(N,0,gpu,re);end
        if (gpu && isaUnderlying(x,'double')) || isa(x,'double');F=double(F);end
        if m~=1
            perm=1:ndims(x);perm([1 m])=[m 1];
            x=permute(x,perm);
        end
        if ~ismatrix(x)
            S=size(x);S(end+1:2)=1;
            x=reshape(x,[S(1) prod(S(2:end))]);
            ism=0;
        end
        %if re==2;x=real(F)*x+1i*(imag(F)*x);else x=F*x;end
        x=F*x;
        if ~ism
            S(1)=size(x,1);
            x=reshape(x,S);
        end
        if m~=1;x=permute(x,perm);end
    end
end