def block_lu(A, r):
    """
    Block LU decomposition using a block size of r.
    """
    n = A.shape[0]

    if n <= r:
        # For small enough matrices, compute LU directly
        L, U = lu_no_pivot(A)
        return L, U

    # Partition A
    A11 = A[:r, :r]
    A12 = A[:r, r:]
    A21 = A[r:, :r]
    A22 = A[r:, r:]

    # Factor A11 = L11U11
    L11, U11 = lu_no_pivot(A11)

    # Solve for U12: L11 U12 = A12
    U12 = np.linalg.solve(L11, A12)

    # Solve for L21: L21 U11 = A21
    L21 = np.linalg.solve(U11, A21.T).T

    # Compute the Schur complement S
    S = A22 - L21 @ U12

    # Recursive LU decomposition on S
    L22, U22 = block_lu(S, r)

    # Form the full L and U from the blocks
    L = np.block([[L11, np.zeros((r, n-r))], [L21, L22]])
    U = np.block([[U11, U12], [np.zeros((n-r, r)), U22]])

    return L, U

# Testing the block LU decomposition function
r = 2
L_block, U_block = block_lu(A_sample, r)
L_block, U_block
