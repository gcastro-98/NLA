

import numpy as np

"""
Standard matrix multiplication

1) Involves 2n^3 flop (counting additions and multiplications separately)
2) Produces and consumes only  3n^2 data values

Problem: data is not reused! for each i (outer loop) the data of A[i,] is not reused any more! 

Idea: Take advantage of this and access small blocks (that can be stored or even move to fast memory acces area).
"""

################################


def blockmatprod(A,B):
    C=np.zeros((n,n)); 
    for i in range (0,N): 
        for j in range (0,N): # <- here we load block C[i,j] into fast  memory
            for k in range (0,N): # <- here we load block A[i,k], B[k,j] into fast memory
                for ii in range (i*b,(i+1)*b):
                    for jj in range (j*b,(j+1)*b):  
                        for kk in range (k*b,(k+1)*b):
                            C[ii,jj]=C[ii,jj]+A[ii,kk]*B[kk,jj];
            # <- here we write block C[ii,jj] into slow memory
    return C;



"""
MAIN

Sup n=Nb, with b=block size small enough so that we can fit one block of A,B and C in Cache Memory (Fast Memory area)
"""
p=7
n=2**p  #dimension (e.g. a power of 2)

A=np.random.rand(n,n); B=np.random.rand(n,n)  #random matrices

C1=np.dot(A,B);  print('Direct computation %d\n' % n);  #direct multiplication of matrices


for pp in range (0,p+1):
     N=2**pp; b=int(n/N); 
     C2=blockmatprod(A,B);  #block matrix multiplication
     print(' %4d %4d %4d %24.16e\n'%(n,b,N,np.linalg.norm(C2-C1,np.inf)))
