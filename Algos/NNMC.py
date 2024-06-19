import numpy as np
from Algos.Helpers import svd_threshold
from copy import deepcopy


        
def NNMC(X, M, Z, Sampled_mask, Nu, epsilon, threshold_lambda, rho=1, beta=1, max_iter=1000, min_iter=1):
    '''
    Non-negative Matrix Completion algorithm with ADMM implementation as described in the paper: doi: 10.1109/MLSP.2013.6661924
    X: matrix to complete
    M: initial guess for the matrix to complete
    Z: initial guess for the low-rank matrix
    Sampled_mask: boolean matrix of the same size as X, with True values where we have sampled (observed) the matrix X. If you use the function sample_from_matrix, you can use the third output of the function
    Nu: initial guess for the Lagrange dual variable
    ep: convergence threshold
    threshold_lambda: threshold value for the singular value thresholding
    rho: penalty parameter for the equality constraint
    beta: (1 or 2) specifies the beta divergence to use. beta=1 corresponds to the Kullback-Leibler divergence, beta=2 corresponds to the Euclidean divergence
    max_iter: maximum number of iterations
    min_iter: minimum number of iterations
    '''
    
    # Y, X, Z numpy arrays
    iteration=0 # initialize iteration counter
    

    while iteration < max_iter:
        
        M_prev = deepcopy(M)
        
        if beta == 1:
            M = np.where(Sampled_mask, ((rho*Z-Nu-1)+np.sqrt((rho*Z-Nu-1)**2+4*rho*X))/(2*rho), np.maximum(Z-1/rho*Nu,0))
        elif beta == 2:
            M = np.where(Sampled_mask, (np.maximum(rho*Z-Nu+X,0))/(1+rho), np.maximum(Z-1/rho*Nu,0))
        
        Z = svd_threshold(M+1/rho*Nu,threshold_lambda/rho)
        
        Nu = Nu + rho*(M-Z)
        
        error_current = np.max(np.abs(M_prev - M))
        
        if error_current < epsilon and iteration >= min_iter:
            print(f'Converged at iteration {iteration} to error {error_current}')
            break
        else:
            print(f'Error at iteration {iteration}: {error_current}')
            iteration += 1
    np.where(M==0,np.nan,M)
    return M