# complexSVDShrinkageDWI
Tools for complex DWI denoising using SVD shrinkage

This repository provides tools to implement the methods and reproduce the experiments included in the manuscript ''Complex diffusion-weighted image estimation via matrix recovery under general noise models'', L Cordero-Grande, D Christiaens, J Hutter, AN Price, and JV Hajnal, arXiv:1812.05954.

The code has been developed in MATLAB and has the following structure:

###### ./
contains the scripts for running the experiments included respectively in Figs. 4, 9, 6 and 5  of the manuscript: *complexSVDShrinkageDWI_Exp[1-4].m*, generate part of the graphical materials included in the Figs.: *plot_Exp[1-4].m*, call the patch-based traversal: *patchSVShrinkage.m* and perform SVD shrinkage: *SVShrinkage.m*.

###### ./Build
contains scripts that replace, extend or adapt some MATLAB built-in functions: *diagm.m*, *dynInd.m*, *eigm.m*, *emtimes.m*, *ind2subV.m*, *indDim.m*, *matfun.m*, *multDimMax.m*, *multDimSum.m*, *numDims.m*, *parUnaFun.m*, *resPop.m*, *resSub.m*, *sub2indV.m*, *svdm.m*.

###### ./ESDEstimation
contains scripts for empirical sample distribution computation: *ESDMixAndMix.m*, *ESDSimulated.m*, *ESDSpectrode.m*.

###### ./ESDEstimation/MixAndMixCode
contains scripts that implement the MidAndMix ESD computation method in the manuscript ''Numerical techniques for the computation of sample spectral distributions of population mixtures'', L Cordero-Grande, arXiv:1812.05575: *andersonMixing.m*, *arraySupport.m*, *fillGridPoints.m*, *gridSubdivide.m*, *interp1GPU.m*, *nonUniformGridAddPoints.m*, *pinvmDamped.m*, *startingGrid.m*.

###### ./ESDEstimation/SpectrodeCode
contains scripts that implement the Spectrode ESD computation method in the manuscript ''Efficient computation of limit spectra of sample covariance matrices'', E Dobriban, Rand. Matr. Th. Appl., 2015, 4(4):1550019:1-36 from https://github.com/dobriban/EigenEdge/

###### ./Figures
contains scripts for generating the graphical materials: *genericFigInformation.m*.

###### ./Lib
contains external MATLAB tools.

###### ./Lib/Colormaps
contains tools to generate colormaps from https://uk.mathworks.com/matlabcentral/fileexchange/51986-perceptually-uniform-colormaps.

###### ./Lib/line_fewer_markers_v4
contains tools for plotting from https://uk.mathworks.com/matlabcentral/fileexchange/42560-line_fewer_markers.

###### ./Lib/NIfTI_20140122
contains tools for tools for NIfTI and ANALYZE formats from https://uk.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image

###### ./Matrices
contains scripts for operations with matrices: *mat2bldiag.m*.

###### ./Methods
contains scripts that implement generic methods for reconstruction: *build1DFTM.m*, *buildStandardDFTM.m*, *fctGPU.m*, *fftGPU.m*, *generateGrid.m*, *ifctGPU.m*, *ifftGPU.m*, *mirroring.m*, *plugNoise.m*, *resampling.m*, *ridgeDetection.m*, *spaNeigh.m*.

###### ./Shrinkage
contains scripts for SVD shrinkage: *frobenius.m*, *generalShrinkage.m*, *hard.m*, *operator.m*, *percMarcenkoPastur.m*, *stieltjes.m*, *stieltjesSimulated.m*, *veraart.m*.


NOTE 1: Exemplary data is provided in the datasets *recFig0[4,6a,6b,9].mat*, *dwi[w,m,x,y].nii*, *dw_scheme_b10000.[bval,bvec]*. For runs without changing the paths, they should be placed in a folder
###### ../complexSVDShrinkageDWIData
Data generated when running the scripts is also stored in this folder as *retFig0[4,5,8].mat* and *dwix[GSVS-stdhat,GSVS,MPPCA].nii*. Script used to run the NLSAM method from https://github.com/samuelstjean/nlsam/releases/download/v0.6.1/nlsam_0.6.1_linux_x64.zip is also included: *nlsam.run.sh*.


NOTE 2: Computation times of the provided execution mode ('Quick' in flag "typExec") on an 8(16) x Intel(R) Core(TM) i7-5960X CPU @ 3.00GHz 64GB RAM with a GeForce GTX TITAN X range from 1' to 5' depending on the denoising method. Note Experiment 1 (Fig. 4) runs seven methods, Experiment 2 (Fig. 9) runs four methods, Experiment 3 (Fig. 6) runs two methods and Experiment 4 (Fig. 5) runs three methods.

