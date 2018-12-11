function C=addCorrelation(C,rho,pow)

%ADDCORRELATION   Adds a given amount of correlation to covariance matrices
%   C=ADDCORRELATION(C,RHO)
%   * C is a set of diagonal covariance matrices
%   * RHO is the Pearson correlation coefficient to be added
%   * C is a set of covariance matrices
%

Cd=diagm(C);
M=size(Cd,2);
Cp=1:M;
C=C+bsxfun(@times,rho.^(abs(bsxfun(@minus,matfun(@transpose,Cp),Cp)).^pow),(sqrt(bsxfun(@times,matfun(@ctranspose,Cd),Cd))-C));

