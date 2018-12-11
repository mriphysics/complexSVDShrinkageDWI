function A=emtimes(A,B)

%EMTIMES   Overrides the behaviour of page matrix multiplication (mtimes) 
%by permuting the arrays so that dimensions maps as D(A):1,2->1,2; 
%D(B):1,2->2,3 (and pages are matched), doing kronecker multiplication, 
%contracting the second dimension and permuting back D(A):1,3->1:2. This
%may be useful to accelerate multiplications of a sequence of small 
%matrices
%   A=EMTIMES(A,B)
%   * A is the first input array
%   * B is the second input array
%   * A is the output array
%
try
    NDA=max(numDims(A)+1,4);NDB=max(numDims(B)+1,4);NDm=max(NDA,NDB);
    permA=[1 2 NDA+1 3:NDA];permB=[NDB+1 1:NDB];permI=[1 3:NDm+1 2];
    A=permute(sum(bsxfun(@times,permute(A,permA),permute(B,permB)),2),permI); 
catch%In case issues with memory
    A=matfun(@mtimes,A,B);
end



