import numpy as np
from Algos.Helpers import k_0_finder, projection_operator, suggested_stop

def SVT(Sampled_matrix, tau=None, epsilon=1e-7, step_size=None, max_iter=100, max_rank=None):
    """
    Singular Value Thresholding (SVT) algorithm for matrix completion. doi: https://doi.org/10.1137/080738970
    Sampled_matrix: the matrix with missing entries
    tau: the thresholding parameter. Higher values of tau will result in a lower rank approximation
    epsilon: the tolerance for the convergence criterion
    step_size: the step size for the gradient descent, default is 1.2/p where p is the fraction of observed entries. If the results blow up, reduce the step size
    max_iter: the maximum number of iterations
    max_rank: the maximum rank of the approximation. If the rank of the approximation exceeds this value, the algorithm will stop
    
    """
    if max_rank is None:
        max_rank = np.max(Sampled_matrix.shape)
    
    if step_size is None:
        m = Sampled_matrix.shape[0]
        n = Sampled_matrix.shape[1]
        p = m/(n*n)
        step_size = 1.2/p
    
    if tau is None:
        n = Sampled_matrix.shape[1]
        tau = 5*(n)
    # Initialize k and a Y_0
    k = 1
    Y_0 = k_0_finder(tau, step_size, Sampled_matrix) * step_size * Sampled_matrix

    # First singular value calculation
    u, s, v = np.linalg.svd(Y_0, full_matrices=False)
    print('The biggest singular value on iteration 0:' , s[0])


    # Get list ready
    iterate = True

    rank_arr = []
    rel_error_s_arr = []

    while iterate:
        print(f'Iter {k}')
        if k == 1:
            Y = Y_0

        # Singular Value Decomposition
        U, sigma, Vt = np.linalg.svd(Y, full_matrices=False)

        # Thresholding
        big_sigma = sigma[0]
        print(f'Biggest singular value = {big_sigma}')
        sigma_thresh = np.maximum(sigma - tau, 0)
        # Check for rank stopping condition
        rank = np.sum(sigma_thresh > 0)
        # Construct the approximation matrix X
        X = U[:, :rank] @ np.diag(sigma_thresh[:rank]) @ Vt[:rank, :]

        # Calculate errors
        error_sampled_matrix = -projection_operator(M_sampled=Sampled_matrix, X=X)
        rel_error_s = (np.linalg.norm(projection_operator(Sampled_matrix, X=X), 'fro') / np.linalg.norm(Sampled_matrix, 'fro'))

        if rel_error_s<1e-2:
            step_size *=.9

        # Update Y
        Y = Y + step_size * error_sampled_matrix

        # Append results for plotting
        rank_arr.append(rank)
        rel_error_s_arr.append(rel_error_s)

        # Check stopping conditions
        if rank > max_rank:
            break
        if suggested_stop(X, Sampled_matrix, epsilon):
            break
        if k >= max_iter:
            break
        k += 1
        print(f'Iter {k-1}; Relative error:', rel_error_s)
        print('rank:',rank)
        print('\n')

    return X, rank_arr, k