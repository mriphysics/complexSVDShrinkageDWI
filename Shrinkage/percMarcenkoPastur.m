function percV=percMarcenkoPastur(beta,perc)

%PERCMARCENKOPASTUR  computes a given percentile of the Marcenko-Pastur
%distribution. It adapts the code provided in [1] M Gavish, DL Donoho, 
%"Optimal shrinkage of singular values," IEEE Trans Inf Theory, 
%63(4):2137-2152, 2017.
%   PERC=PERCMARCENKOPASTUR(BETA,{PERC})
%   * BETA is the shape factor of the random matrix
%   * {PERC} is the percentile of the distribution we want to extract.
%   Defaults to 0.5.
%   * PERCV is the value of the distribution at that percentile
%

if nargin<2 || isempty(perc);perc=0.5;end

MarPas=@(x)1-incMarPas(x,beta,0);
lobnd=(1-sqrt(beta))^2;
hibnd=(1+sqrt(beta))^2;
change=1;
while change && (hibnd-lobnd>.001)
    change=0;
    x=linspace(lobnd,hibnd,5);
    y=zeros(size(x));
    for n=1:length(x);y(n)=MarPas(x(n));end  
    if any(y<perc);lobnd=max(x(y<perc));change=1;end
    if any(y>perc);hibnd=min(x(y>perc));change=1;end
end
percV=(hibnd+lobnd)./2;

end

function I=incMarPas(x0,beta,gamma)

assert(beta<=1,'Matrix shape factor should be lower or equal than 1 and it is',beta);
topSpec=(1+sqrt(beta))^2;
botSpec=(1-sqrt(beta))^2;
MarPas=@(x)IfElse((topSpec-x).*(x-botSpec) >0,sqrt((topSpec-x).*(x-botSpec))./(beta.* x)./(2 .* pi),0);
if gamma~=0;fun=@(x)(x.^gamma.* MarPas(x));else fun=@(x)MarPas(x);end
I=integral(fun,x0,topSpec);

function y=IfElse(Q,point,counterPoint)
    y = point;
    if any(~Q)
        if length(counterPoint)==1;counterPoint=ones(size(Q)).*counterPoint;end
        y(~Q)=counterPoint(~Q);
    end        
end

end