import numpy as np
import scipy.linalg

def lunopiv(A, n, ptol: float = 1e-3):
	for k in range(n-1):
		pivot = A[k, k]
		if abs(pivot) < ptol:
			print("Zero pivot encountered")
			break

		for i in range(k+1, n):
			A[i, k] = A[i, k] / pivot
			for j in range(k+1, n):
				A[i, j] = A[i, j] - A[i, k] * A[k, j]
	L = np.eye(n) + np.tril(A, -1)
	U = np.triu(A)
	return L, U


# Testing the function with a sample matrix
A_sample = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
L_sample, U_sample = lunopiv(A_sample, n)
print(L_sample, U_sample)