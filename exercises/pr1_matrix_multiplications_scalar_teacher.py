

import numpy as np


"""
ijk -> the inner loop is an inner product of two vectors (a dot product), 
                 requires acces to a row of A and a column of B

       the middle loop performs a (vector x matrix) operation 
                 requires acces to a row of A and the matrix B
"""

def ijk(A,B,C):
   for i in range (0,n):
       for j in range (0,n):
           for k in range (0,n):
               C[i,j] = A[i,k]*B[k,j] + C[i,j]
   return C


def ijk_improved(A,B,C):
   for i in range (0,n):
       for j in range (0,n):
           C[i,j] = np.dot(A[i,:],B[:,j]) + C[i,j]
   return C


"""
jik -> similar to ijk, the inner loop is also an inner/dot product
                 requires acces to a row of A and a column of B

       the middle loop performs a (matrix x vector) operation
                 requires acces to the matrix A and a column of B
"""
def jik(A,B,C):
   for j in range (0,n):
       for i in range (0,n):
           for k in range (0,n):
               C[i,j] = A[i,k]*B[k,j] + C[i,j]
   return C


"""
ikj -> the inner loop is the multiplication of a number by a row of B (saxpy)
       the middle loop is row gaxpy operation 
"""

def ikj(A,B,C):
   for i in range (0,n):
       for k in range (0,n):
           for j in range (0,n):
               C[i,j] = A[i,k]*B[k,j] + C[i,j]
   return C


def ikj_improved(A,B,C):
   for i in range (0,n):
       for k in range (0,n):
           aux = A[i,k]               # only one memory access!
           for j in range (0,n):
               C[i,j] = aux*B[k,j] + C[i,j]
   return C


"""
jki -> inner loop (saxpy)
       middle loop (column gaxpy)
       similar to ikj
"""

def jki(A,B,C):
   for j in range (0,n):
       for k in range (0,n):
           for i in range (0,n):
               C[i,j] = A[i,k]*B[k,j] + C[i,j]
   return C
        

"""
kij -> inner loop (saxpy)
       middle loop (row outer product or tensorial product) 

       Rec: Given two vectors, a = [a0, a1, ..., aM] and b = [b0, b1, ..., bN], the outer product is the matrix

            [[a0*b0  a0*b1 ... a0*bN ]
             [a1*b0    .
             [ ...          .
             [aM*b0            aM*bN ]]


"""
def kij(A,B,C):
   for k in range (0,n):
       for i in range (0,n):
           for j in range (0,n):
               C[i,j] = A[i,k]*B[k,j] + C[i,j]
   return C
 
def kij_improved(A,B,C):
   for k in range (0,n):
       C = np.outer(A[:,k],B[k,:]) + C
   return C
 

"""
kji -> similar to kij
       middle loop (column outer product)
"""    
def kji(A,B,C):
   for k in range (0,n):
       for j in range (0,n):
           for i in range (0,n):
               C[i,j] = A[i,k]*B[k,j] + C[i,j]
   return C
 


"""
MAIN
"""

n=5

A=np.random.rand(n,n)
B=np.random.rand(n,n)
C=np.zeros((n,n))

print(A); print("\n")
print(B); print("\n")
print(C); print("\n")

C=kij_improved(A,B,C)
print(C); print("\n")
print(np.dot(A,B))
