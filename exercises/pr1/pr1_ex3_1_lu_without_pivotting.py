import numpy as np

def lu_no_pivot(A):
    """
    Compute the LU factorization without pivoting for a square matrix A.
    A = LU
    """
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    
    for k in range(n):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        if k < n - 1:
            L[k+1:, k] = (A[k+1:, k] - L[k+1:, :k] @ U[:k, k]) / U[k, k]
    
    return L, U

# Testing the function with a sample matrix
A_sample = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
L_sample, U_sample = lu_no_pivot(A_sample)
L_sample, U_sample
