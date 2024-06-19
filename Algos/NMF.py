import numpy as np
from Algos.Helpers import general_error_relative

def NMF(X, rank, epsilon=1e-4, max_iter=300, min_iter=1, seed=42):
    '''
    Non-negative matrix factorization algorithm (Multiplicative Update Rules) as described in the paper: doi: https://doi.org/10.1038/44565
    X: the matrix to factorize
    rank: the rank of the factorization
    epsilon: the relative error threshold to stop the algorithm
    max_iter: the maximum number of iterations
    min_iter: the minimum number of iterations
    seed: the seed for the random number generator
    returns: W, H the factorized matrices, W: m x rank, H: rank x n
    '''
    # initialize the random number generator with the seed
    rng = np.random.default_rng(seed)
    
    m, n = X.shape
    
    W = rng.random((m,rank))
    H = rng.random((rank,n))
    iteration = 0
    
    while iteration < max_iter and iteration >= min_iter:
        
        H *= (W.T@X)/(W.T@W@H) # update rule for H
        W *= (X@H.T)/(W@H@H.T) # update rule for W
        current_error = general_error_relative(X, W@H)
        if current_error < epsilon:
            return W,H
        else:
            print(f'General relative error at iteration {iteration}: {current_error}')
        iteration += 1
        
    return W, H