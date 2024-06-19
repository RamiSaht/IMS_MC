import numpy as np

def SVD_truncated(A, k):
    """
    Truncate the SVD of A to keep only the k largest singular values
    """
    
    U, S, V = np.linalg.svd(A, full_matrices=False)
    
    n= len(S)
    
    if k > n:
        k = n
    
    s= s[:k]
    U = U[:, :k]
    V = V[:k, :]
    return U, s, V
