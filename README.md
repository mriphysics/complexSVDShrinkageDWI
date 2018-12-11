# complexSVDShrinkageDWI
Tools for complex DWI denoising using SVD shrinkage

This repository provides tools to implement the methods and reproduce the experiments included in the manuscript ''Complex diffusion-weighted image estimation via matrix recovery under general noise models'', L Cordero-Grande, D Christiaens, J Hutter, AN Price, and JV Hajnal. Unpublished.

The code has been developed in MATLAB and has the following structure:

###### ./
contains the scripts for running the experiments included respectively in Figs. 4, 8 and 5 of the manuscript: *complexSVDShrinkageDWI_Exp[1-3].m*, generate part of the graphical materials included in the Figs.: *plot_Exp[1-3].m*, call the patch-based traversal: *patchSVShrinkage.m* and perform SVD shrinkage: *SVShrinkage.m*.

###### ./Build
contains scripts that replace, extend or adapt some MATLAB built-in functions: *diagm.m*, *dynInd.m*, *eigm.m*, *emtimes.m*, *ind2subV.m*, *matfun.m*, *multDimMax.m*, *numDims.m*, *parUnaFun.m*, *resPop.m*, *resSub.m*, *sub2indV.m*, *svdm.m*.

###### ./ESDEstimation
contains scripts for empirical sample distribution estimation: *ESDMixAndMix.m*, *ESDSimulated.m*, *ESDSpectrode.m*.

###### ./ESDEstimation/MixAndMixCode
contains scripts that implement the MidAndMix ESD estimation method in the manuscript ''Numerical techniques for the computation of sample spectral distributions of population mixtures'', L Cordero-Grande. Unpublished: *addCorrelation.m*, *andersonMixing.m*, *arraySupport.m*, *fillGridPoints.m*, *gridSubdivide.m*, *interp1GPU.m*, *nonUniformGridAddPoints.m*, *pinvmDamped.m*, *plotObjective.m*, *startingGrid.m*.

###### ./ESDEstimation/SpectrodeCode
contains scripts that implement the Spectrode ESD estimation method in the manuscript ''Efficient computation of limit spectra of sample covariance matrices'', E Dobriban, Rand. Matr. Th. Appl., 2015, 4(4):1550019:1-36 from https://github.com/dobriban/EigenEdge/

###### ./Figures
contains scripts for generating the graphical materials: *genericFigInformation.m*.

###### ./Figures/Colormaps
contains tools to generate colormaps from https://uk.mathworks.com/matlabcentral/fileexchange/51986-perceptually-uniform-colormaps.

###### ./Figures/line_fewer_markers_v4
contains tools for plotting from https://uk.mathworks.com/matlabcentral/fileexchange/42560-line_fewer_markers.

###### ./Matrices
contains scripts for operations with matrices: *mat2bldiag.m*.

###### ./Methods
contains scripts that implement generic reconstruction methods: *build1DFTM.m*, *buildStandardDFTM.m*, *fftGPU.m*, *generateGrid.m*, *ifftGPU.m*, *plugNoise.m*, *resampling.m*, *ridgeDetection.m*, *spaNeigh.m*.

###### ./Shrinkage
contains scripts for SVD shrinkage: *frobenius.m*, *generalShrinkage.m*, *hard.m*, *operator.m*, *percMarcenkoPastur.m*, *stieltjes.m*, *stieltjesSimulated.m*, *veraart.m*.


NOTE 1: Exemplary data is provided in the datasets *recFig0[4,5,8].mat*. For runs without changing the paths, they should be placed in a folder
###### ../complexSVDShrinkageDWIData
Data generated when running the scripts is also stored in this folder as *retFig0[4,5,8].mat*.

NOTE 2: Computation times of the provided execution mode ('Quick' in flag "typExec") on an 8(16) x Intel(R) Core(TM) i7-5960X CPU @ 3.00GHz 64GB RAM with a GeForce GTX TITAN X range from 1' to 5' depending on the denoising method. Note Experiment 1 (Fig. 4) runs seven methods, Experiment 2 (Fig. 8) runs four methods and Experiment 3 (Fig. 5) runs two methods.

