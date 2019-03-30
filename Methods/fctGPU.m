function x=fctGPU(x,m,gpu,F)

%FCTGPU   Configurable GPU-based FCT computation
%   X=FCTGPU(X,M,GPU,{F})
%   * X is the array on which to apply the FCT (Fast Cosine Transform)
%   * M is the direction along which to apply the FCT
%   * GPU is a flag that determines whether to use cpu/gpu with built-in
%   matlab functions (0) / kernel gpu (1), kernel matrix-based gpu (2), gpu
%   (3) or matrix-based gpu computation (4). Some modifications are 
%   introduced to the user preferences due to bugs in gpu computations in 
%   matlab 2015a (roughly speaking these affected the gpu computation of 
%   the fft for certain ---usually prime numbers--- sizes of the array when 
%   the computation was not performed along the first dimension), so users
%   should better test the matlab gpu fft and dct behaviour in their 
%   systems with with different array sizes and along different dimensions
%   * {F} is a FCT matrix (or any other square matrix) provided by the 
%   user. If not provided and required, it is obtained by the function
%   * X is the FCT-transformed array
%

if nargin<4;F=[];end
if gpu==1;gpu=gpu+2;end%Kernel method based on fft not implemented for this transform
if strcmp(version('-release'),'2015a')
    if gpu==2;gpu=gpu+2;end
    if gpu==0;gpu=3;end
end  

ism=1;
if gpu==0
    x=fct(x,[],m);
elseif gpu==3 || gpu==1
%elseif gpu==3 || (gpu==1 && ndims(x)~=3)
    if m==1
        if size(x,1)~=1;x=fct(x);end
    else
        perm=1:ndims(x);perm(1)=m;perm(m)=1;
        x=permute(x,perm);
        if size(x,1)~=1;x=fct(x);end
        x=permute(x,perm);
    end
elseif gpu==4 || gpu==2
%elseif gpu==4
    N=size(x,m);
    if N~=1
        if isempty(F);F=gpuArray(single(dctmtx(N)));end
        if m~=1
           perm=1:ndims(x);perm([1 m])=[m 1];
           x=permute(x,perm);
        end
        if ~ismatrix(x)
            S=size(x);S(end+1:2)=1;
            x=reshape(x,[S(1) prod(S(2:end))]);
            ism=0;
        end
        x=F*x;
        if ~ism;x=reshape(x,S);end
        if m~=1;x=permute(x,perm);end
    end
%Currently not used from here on
elseif gpu==1
    N=size(x,m);
    if N~=1;x=fft_kernel(complex(x),m);end
elseif gpu==2
    N=size(x);N(end+1:4)=1;
    if N(m)~=1
        if isempty(F);F=gpuArray(complex(single(dctmtx(N(m)))));elseif size(F,2)~=N(m);error('Number of columns of the matrix (%d) does not match array size (%d)',size(F,2),N(m));end
        x=reshape(x,[N(1:3) prod(N(4:end))]);
        x=fft_kernel(complex(x),m,F);
        x=reshape(x,N);
    end
end