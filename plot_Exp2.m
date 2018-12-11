%PLOT_EXP1 script generates part of Fig. 2 of manuscript ''Complex 
%diffusion-weighted image estimation via matrix recovery under general 
%noise models'', L. Cordero-Grande, D. Christiaens, J. Hutter, A.N. Price, 
%and J.V. Hajnal

addpath(genpath('.'));
pathData='../complexSVDShrinkageDWIData';
load(fullfile(pathData,'recFig08.mat'),'y');
load(fullfile(pathData,'retFig08.mat'),'xv');

genericFigInformation;

%FIGURE8
y={y,xv{2,2},xv{2,1},xv{1,2},xv{1,1}};
NY=length(y);

for n=1:NY
    y{n}=dynInd(abs(y{n}),2,4);
    y{n}=permute(gather(dynInd(y{n},{3:63,16:88,ceil(size(y{n},3)/2-15)},[1 2 3])),[1 4 2 3]);
end
z=cat(4,y{:});

s0={''};
indShow=[1];
s0=s0(indShow);

meth={'\textbf{a) Original}','\textbf{b) Magn. Veraart}','\textbf{c) Magn. Ours}','\textbf{d) Comp. Veraart}','\textbf{e) Comp. Ours}'};
z=bsxfun(@rdivide,z,multDimMax(z(:,:,:,2:end),[1 3:4]));
z(z>1)=1;
z=padarray(z,[1 0 1 0],1+1/256);

N=size(z);
z=dynInd(z,indShow,2);
N=size(z);N(end+1:4)=1;
z=permute(z,[3 2 1 4]);
N=size(z);
z=reshape(z,[prod(N(1:2)) prod(N(3:4))]);
NZ1=prod(N(1:2));
NZ2=prod(N(3:4));
for s=1:length(indShow)
    figure
    imshow(sqrt(z),sqrt([0 multDimMax(z(:,4*NZ2/5+1:NZ2),1:2)]),'Border','tight','InitialMagnification',200,'Border','tight')%Sqrt to improve contrast

    set(gca, 'Units', 'Pixels');
    pogca = get(gca, 'Position');
    set(gca,'fontsize',FontSizeC)
    set(gcf,'InvertHardCopy','off');

    pos=pogca(3:4);
    fig=gcf; 
    fig.Color=[1 1 1];
    fig.Position=[1 1 pos(1)+37 pos(2)+24];

    N=pos(1)/10;
    sh=[32 13 21 13 21];
    for n=1:5;text('units','pixels','position',[(n-1)*2*N+sh(n) pos(2)+8],'string',meth{n},'Interpreter','latex','Fontsize',FontSizeD-8);end
end