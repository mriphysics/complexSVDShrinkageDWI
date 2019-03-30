%PLOT_EXP1 script generates Fig. 4 of manuscript ''Complex 
%diffusion-weighted image estimation via matrix recovery under general 
%noise models'', L. Cordero-Grande, D. Christiaens, J. Hutter, A.N. Price, 
%and J.V. Hajnal

clearvars
curFolder=fileparts(mfilename('fullpath'));
addpath(genpath(curFolder));%Add code
pathData=strcat(curFolder,'/../complexSVDShrinkageDWIData');%Data path
load(fullfile(pathData,'recFig04.mat'),'UpsilonInv');
load(fullfile(pathData,'retFig04.mat'),'sigmav','Rv');

genericFigInformation;

%FIGURE4A
y={UpsilonInv,sigmav{1,2},sigmav{1,3},sigmav{1,4},sigmav{2,2},sigmav{2,3},sigmav{2,4}};
NY=length(y);
y{1}=y{1}>1e-3;%Mask

figure
for s=2:NY
    y{s}=y{s}(y{1});
    [f,x]=ksdensity(gather(y{s}),'function','cdf');    
    if s<5;line_fewer_markers(x,f,10,'LineWidth',LineWidth,'Color',co(s-1,:),'LineStyle','--','Marker',markers{s-1});
    else line_fewer_markers(x,f,10,'LineWidth',LineWidth,'Color',co(s-1,:),'LineStyle',':','Marker',markers{s-1});
    end
    hold on
end
set(gca,'fontsize',FontSizeC)
title('\textbf{a) Cumulative densities of noise estimates}','FontSize',FontSizeA-4,'Interpreter','latex')
xlabel('$\hat{\sigma}$','FontSize',FontSizeA,'Interpreter','latex')
ylabel('$F(\hat{\sigma})$','FontSize',FontSizeA,'Interpreter','latex','rotation',0,'HorizontalAlignment','right')
grid on
legend({'$\hat{\sigma}^{\mathbf{Y}}_{\mathrm{EXP1}}$','$\hat{\sigma}^{\mathbf{Y}}_{\mathrm{EXP2}}$','$\hat{\sigma}^{\mathbf{Y}}_{\mathrm{MED}}$','$\hat{\sigma}^{\mathbf{W}}_{\mathrm{EXP1}}$','$\hat{\sigma}^{\mathbf{W}}_{\mathrm{EXP2}}$','$\hat{\sigma}^{\mathbf{W}}_{\mathrm{MED}}$'},'Interpreter','latex','FontSize',FontSizeA,'Location','NorthWest');
xlim([0.4 1.4])

set(gcf,'InvertHardCopy','off');
fig=gcf;
fig.Position=[1 1 1920 1200];
fig.Color=[1 1 1];

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1)+0.04;
bottom = outerpos(2) + ti(2)+0.03;
ax_width = outerpos(3) - ti(1) - ti(3)-0.04;
ax_height = outerpos(4) - ti(2) - ti(4)-0.03;
ax.Position = [left bottom ax_width ax_height];

%FIGURE4B
y={sigmav{2,3},sigmav{1,3},Rv{1,3},Rv{1,1}};

NY=length(y);
minmax=zeros(NY,2);
for n=1:NY
    y{n}=gather(dynInd(y{n},ceil(size(y{n},3)/2-7),3));
    if n>=3;y{n}=100*y{n}/550;end
    minmax(n,:)=[min(y{n}(:)) max(y{n}(:))];
end

pause(1)

for n=1:NY
    figure
    imshow(abs(y{n}'),[])
    set(gca, 'Units', 'normalized', 'Position', [0 0 0.9 0.85])
    set(gca,'fontsize',FontSizeB)
    set(gcf,'InvertHardCopy','off');

    fig=gcf; 
    fig.Color=[1 1 1];
    fig.Position=[1 1 1200 1200];%get(0,'Screensize');%Screensize=[1 1 1920 1200]

    if n<3
        rr=[min(minmax(1:2,1));max(minmax(1:2,2))];       
        caxis(rr);
        colormap('viridis');
    else
        rr=[min(minmax(3:4,1));max(minmax(3:4,2))];        
        newmap=colormap('inferno');
        newmap=histeq(gray2ind(mat2gray([abs(y{3}) abs(y{4})],double(rr)),256),newmap);
        caxis(rr);
        colormap(newmap);
    end
    cb=colorbar;
    colorTitleHandle = get(cb,'Title');
    cb.FontSize=FontSizeA;
    if n==1;title('\textbf{$\quad$b) $\hat{\sigma}^{\mathbf{W}}_{\mathrm{EXP2}}$ (noise only)}','FontSize',FontSizeA+26,'Interpreter','latex');
    elseif n==2;title('\textbf{$\quad$c) $\hat{\sigma}^{\mathbf{Y}}_{\mathrm{EXP2}}$ (signal plus noise)}','FontSize',FontSizeA+26,'Interpreter','latex');
    elseif n==3;title('\textbf{$\quad$d) $\hat{R}^{\mathbf{Y}}_{\mathrm{EXP2}}\,(\%)$ (EXP2)}','FontSize',FontSizeA+26,'Interpreter','latex');
    else title('\textbf{$\quad$e) $\hat{R}^{\mathbf{Y}}\,(\%)$ (ours)}','FontSize',FontSizeA+26,'Interpreter','latex');
    end
    cb.Position(4)=0.8;
    cb.Position(2)=0.015;
    cb.Position(1)=0.85;
end
