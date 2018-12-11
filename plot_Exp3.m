%PLOT_EXP3 script generates part of Fig. 5 of manuscript ''Complex 
%diffusion-weighted image estimation via matrix recovery under general 
%noise models'', L. Cordero-Grande, D. Christiaens, J. Hutter, A.N. Price, 
%and J.V. Hajnal

addpath(genpath('.'));
pathData='../complexSVDShrinkageDWIData';
load(fullfile(pathData,'recFig05.mat'),'y','voxsiz');
load(fullfile(pathData,'retFig05.mat'),'xv');

genericFigInformation;

%FIGURE5
y={y,xv{2,1},xv{1,1}};
NY=length(y);

minmax=zeros(2,NY);
for n=1:NY
    y{n}=gather(dynInd(abs(y{n}),{ceil(size(y{n},3)/2+2),2},3:4));
    y{n}=y{n}';
    y{n}=flip(y{n},2);
        
    y{n}=dynInd(y{n},{20:82,16:89},1:2);
    minmax(:,n)=[min(y{n}(:));max(y{n}(:))]; 
end
minmax(1,1)=min(minmax(1,:),[],2);
minmax(2,1)=max(minmax(2,:),[],2);

titstr={'Original','Denoised NPC','Denoised PC'};

for n=1:NY
    figure 
    imshow(abs(y{n})',[minmax(1,1) minmax(2,1)],'InitialMagnification',400*voxsiz(1),'Border','tight')
    hold on
    set(gca, 'Units', 'Pixels');
    pogca = get(gca, 'Position');
    set(gca,'fontsize',FontSizeC);
    set(gcf,'InvertHardCopy','off');

    fig=gcf; 
    fig.Color=[1 1 1];
    fig.Position=[1 1 378 444+24];
    pause(1)
    title(sprintf('\\textbf{%s}',titstr{n}),'FontSize',FontSizeA-28,'Interpreter','latex');
    pause(1)
end


