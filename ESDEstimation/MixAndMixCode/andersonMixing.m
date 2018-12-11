function [fg,v]=andersonMixing(fg,v,M,w)

%ANDERSONMIXING   Performs an Anderson mixing iteration according to [1] DG 
%Anderson, "Iterative procedures for nonlinear integral equations," J ACM, 
%12:537-560, 1965. The notation is taken from [2] HM Walker and P Ni, 
%"Anderson acceleration for fixed point iterations," SIAM J Num Anal,
%49(3-4):1715-1735, 2011. For systems of equations we rely on some of the
%formulation and numerical technicalities described in [3] PP Pratapaa et
%al, "Anderson acceleration of the Jacobi iterative method: An efficient 
%alternative to Krylov methods for large, sparse linear systems," J Comp
%Phys, 306:43-54, 2016.
%   [FG,V]=ANDERSONMIXING(FG,V,M,{W})
%   * FG is an array storing both the results of the fixed point map (2nd
%   element 1st dimension) and its update (1st element 1st dimension) over
%   previous iterations (2nd dimension)
%   % V is the current estimate of the map
%   * M is the number of previous iterations used for updating the result
%   * {W} is the damping parameter. It defaults to 0.1
%   * FG is the updated array with the results of the fixed point and its
%   update over previous iterations
%   % V is an updated estimate of the map
%

if nargin<4 || isempty(w);w=0.1;end

N=size(fg);
b=fg(2,N(2),:,:)-v;
fg(1,N(2),:,:)=b;
if M>0
    fgu=fg(:,N(2)-M:N(2),:,:);
    F=fgu(1,:,:,:);
    F=diff(F,1,2);   
    [F,b]=parUnaFun({F,b},@permute,[4 2 3 1]);%This way we link the different equations together
    %[F,b]=parUnaFun({F,b},@double);
    F=pinvmDamped(F,w,b);%%%THIS TOLERANCE MAY BE ARGUABLE FOR VERY HIGH ACCURACY (PROBABLY FOR ACCURACIES APPROACHING THAT VALUE, WHICH WE ARE NOT INTERESTED IN)
    F=permute(F,[3 1 2]);     
    NF=size(F);
    al=zeros([NF(1) NF(2)+2],'like',F);al(:,2:end-1)=F;al(:,end)=1;
    al=permute(al,[3 2 1]);
    al=diff(al,1,2);
    v=sum(bsxfun(@times,al,fgu(2,:,:,:)),2);    
else
    v=fg(2,N(2),:,:);
end
fg=fg(:,[2:end 1],:,:);%Much quicker than circshift...
