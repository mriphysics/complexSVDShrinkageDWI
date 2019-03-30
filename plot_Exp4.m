%PLOT_EXP4 script generates part of Fig. 5 of manuscript ''Complex 
%diffusion-weighted image estimation via matrix recovery under general 
%noise models'', L. Cordero-Grande, D. Christiaens, J. Hutter, A.N. Price, 
%and J.V. Hajnal

clearvars
gpu=(gpuDeviceCount>0);%0->Use CPU / 1->Use GPU
if gpu;dev=gpuDevice;end

curFolder=fileparts(mfilename('fullpath'));
addpath(genpath(curFolder));%Add code
pathData=strcat(curFolder,'/../complexSVDShrinkageDWIData');%Data path

%COMPARED METHODS
%In case NLSAM is available
%strMet={'xGSVS','xGSVS-stdhat','xMPPCA','NLSAM','y'};
%strMetLeg={'GSVS','GSVS$\hat{\sigma}$','MPPCA','NLSAM','Original'};
strMet={'xGSVS','xGSVS-stdhat','xMPPCA','y'};
strMetLeg={'GSVS','GSVS$\hat{\sigma}$','MPPCA','Original'};

%DIFFUSION VALUES
diInfo=load(sprintf('%s/dw_scheme_b10000.bval',pathData));
NC=length(strMet);
bval=diInfo(:);
bvalun=sort(unique(bval));
NB=length(bvalun);

%LOAD DATA
nii=load_untouch_nii(sprintf('%s/dwix.nii',pathData));%GROUND-TRUTH
x=single(nii.img);
if gpu;x=gpuArray(x);end
N=size(x);
xM=reshape(x,prod(N(1:3)),[]);
nii=load_untouch_nii(sprintf('%s/dwim.nii',pathData));%MASK
M=single(nii.img);
if gpu;M=gpuArray(M);end
M=M>0.5;
M=reshape(M,prod(N(1:3)),[]);
xM=xM(M,:);
for n=1:NC%DENOISED IMAGES
    nii=load_untouch_nii(sprintf('%s/dwi%s.nii',pathData,strMet{n}));
    xhat{n}=single(nii.img);
    if gpu;xhat{n}=gpuArray(xhat{n});end
    xhatM{n}=reshape(xhat{n},prod(N(1:3)),[]);
    xhatM{n}=xhatM{n}(M,:);
end

%QUALITY COMPUTATIONS
PSNR=zeros([NB NC],'like',x);
SSIM=zeros([NB NC]);
for b=1:NB   
    xb=xM(:,bval==bvalun(b));
    xb=xb(:);
    xhatb=cell(1,NC);
    for n=1:NC
        xhatb{n}=xhatM{n}(:,bval==bvalun(b));
        xhatb{n}=xhatb{n}(:);
    end
    xhatb=cat(2,xhatb{:});
    PSNR(b,:)=max(xb)^2./mean(bsxfun(@minus,xhatb,xb).^2,1);

    xb=x(:,:,:,bval==bvalun(b));
    xhatb=cell(1,NC);
    for n=1:NC;xhatb{n}=xhat{n}(:,:,:,bval==bvalun(b));end
    NV=size(xb,4);
    ssimV=zeros(size(xb,4),1);
    for n=1:NC        
        for l=1:NV;ssimV(l)=ssim(gather(xhatb{n}(:,:,:,l)),gather(xb(:,:,:,l)));end
        SSIM(b,n)=mean(ssimV);
    end        
end
PSNR=10*log10(PSNR);

genericFigInformation;
%FIGURE5a
figure
for sn=1:1
    for s=1:NC
        if sn==1
            plot(bvalun/1000,PSNR(:,s,sn),'Color',co(s,:),'Marker',markers{s},'LineWidth',2);   
        else
            plot(bvalun/1000,PSNR(:,s,sn),'--','Color',co(s,:),'Marker',markers{s},'LineWidth',2); 
        end
        hold on
    end
end
set(gca,'fontsize',FontSizeC)

title(sprintf('\\textbf{a) Errors of different estimators}'),'FontSize',FontSizeA+4,'Interpreter','latex')
xlabel('$b$ (ms/$\mu$m)','FontSize',FontSizeA,'Interpreter','latex')
ylabel({'PSNR';'(dB) '},'FontSize',FontSizeA,'Interpreter','latex','rotation',0,'HorizontalAlignment','right')
grid on
legend(strMetLeg,'Interpreter','latex','FontSize',FontSizeA,'Location','NorthEast');

set(gcf,'InvertHardCopy','off');
fig=gcf;
fig.Position=get(0,'Screensize');
fig.Color=[1 1 1];
fig.PaperPositionMode = 'auto';

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1)+0.04;
bottom = outerpos(2) + ti(2)+0.03;
ax_width = outerpos(3) - ti(1) - ti(3)-0.04;
ax_height = outerpos(4) - ti(2) - ti(4)-0.03;
ax.Position = [left bottom ax_width ax_height];

fig_pos=fig.PaperPosition;
fig.PaperSize=[fig_pos(3) fig_pos(4)];

pause(1)

figure
for sn=1:1
    for s=1:NC
        if sn==1
            plot(bvalun/1000,SSIM(:,s,sn),'Color',co(s,:),'Marker',markers{s},'LineWidth',2);   
        else
            plot(bvalun/1000,SSIM(:,s,sn),'--','Color',co(s,:),'Marker',markers{s},'LineWidth',2);   
        end
        hold on
    end
end
set(gca,'fontsize',FontSizeC)
title(sprintf('\\textbf{b) Perceptual quality of different estimators}'),'FontSize',FontSizeA+4,'Interpreter','latex')
xlabel('$b$ (ms/$\mu$m)','FontSize',FontSizeA,'Interpreter','latex')
ylabel('SSIM','FontSize',FontSizeA,'Interpreter','latex','rotation',0,'HorizontalAlignment','right')
grid on
legend(strMetLeg,'Interpreter','latex','FontSize',FontSizeA,'Location','SouthWest');

set(gcf,'InvertHardCopy','off');
fig=gcf;
fig.Position=get(0,'Screensize');
fig.Color=[1 1 1];
fig.PaperPositionMode = 'auto';

ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1)+0.04;
bottom = outerpos(2) + ti(2)+0.03;
ax_width = outerpos(3) - ti(1) - ti(3)-0.04;
ax_height = outerpos(4) - ti(2) - ti(4)-0.03;
ax.Position = [left bottom ax_width ax_height];

fig_pos=fig.PaperPosition;
fig.PaperSize=[fig_pos(3) fig_pos(4)];

pause(1)