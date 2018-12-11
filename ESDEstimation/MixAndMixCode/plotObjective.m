function plotObjective(eigv,Beta,gridz,range,N)

%PLOTOBJECTIVE  plots the squared residuals of the objective function driven
%by the fixed point equation
%   PLOTOBJECTIVE(EIGV,BETA,GRIDX,{RANGE},{N})
%   * EIGV is an array containing the eigenvalues of the population (assumed 
%   equally probable), it should be a column vector
%   * BETA is the aspect ratio
%   * GRIDZ is the point in the complex domain for which we are interested to
%   observe the residuals
%   * {RANGE} is the area on which we are interested to observe the residuals
%   * {N} is the number of points where to compute the residuals
%

if nargin<4 || isempty(range);range=[-1 1;-1 1];end
if nargin<5 || isempty(N);N=256;end

if numel(N)==1;N=N*ones(1,2);end
eigv=permute(eigv,[2 3 1]);

vgridr=linspace(range(1,1),range(1,2),N(1));
vgridi=1i*linspace(range(2,1),range(2,2),N(2));
[vgrid1,vgrid2]=ndgrid(vgridr,vgridi);
vgridz=vgrid1+vgrid2;
f1=vgridz-mean(bsxfun(@rdivide,eigv,bsxfun(@minus,bsxfun(@rdivide,eigv,1+Beta*vgridz),gridz)),3);

r1=log10(f1.*conj(f1));
v=-10:0.5:10;

figure
surface(vgrid1,imag(vgrid2),r1);
hold on
contour3(vgrid1,imag(vgrid2),r1,v,'k');
xlabel('$\Re\{e\}$','Interpreter','latex')
ylabel('$\Im\{e\}$','Interpreter','latex')
colormap('default') 
colorbar
%grid on
view(2)
set(gcf, 'Position', get(0,'Screensize'))

% %THIS IS TO SEE HOW A FIXED POINT ALGORITHM MAY STILL THEORETICALLY CONVERGE (ALTHOUGH, AS SHOWN IN EXP1, IN AN UNFEASIBLE MANNER)
% ff=-f1./abs(f1);
% figure
% quiver(vgrid1,imag(vgrid2),real(ff),imag(ff))

% %THIS WOULD CHECK THAT ANALOGOUS INTERFERENT BEHAVIOUR IS OBSERVED WITH MHAT
% 
% % if nargin<4;range=[-1 2;-1 1];end
% % 
% % vgridr=linspace(range(1,1),range(1,2),N(1));
% % vgridi=1i*linspace(range(2,1),range(2,2),N(2));
% % [vgrid1,vgrid2]=ndgrid(vgridr,vgridi);
% % vgridz=vgrid1+vgrid2;
% f2=vgridz-(-gridz+Beta*mean(bsxfun(@rdivide,eigv,1+bsxfun(@times,eigv,vgridz)),3)).^(-1);
% r2=log10(f2.*conj(f2));
% 
% figure
% surface(vgrid1,imag(vgrid2),r2);
% xlabel('$\Re\{\tilde{m}\}$','Interpreter','latex')
% ylabel('$\Im\{\tilde{m}\}$','Interpreter','latex')
% colormap('default') 
% colorbar
% grid on
% view(2)