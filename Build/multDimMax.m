function x=multDimMax(x,dim)

%MULTDIMMAX   Takes the maximum of the elements of a multidimensional array along a set of dimensions
%   X=MULTDIMMAX(X,DIM)
%   * X is an array
%   * DIM are the dimensions over which to take the max of the elements of the array
%   * X is the contracted array
%

for n=1:length(dim);x=max(x,[],dim(n));end