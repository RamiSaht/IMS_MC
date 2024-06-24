from Algos.Helpers import OGM_calculate_objective_function, OGM_H, OGM_W, OGM_calculate_objective_function_observed, OGM_calculate_objective_function_unobserved, calculate_sparsity

def OGM_NeNMF(X, Wt, Ht, E, Sampled_mask, beta=8, max_iter=20, epsilon_h=0.01, epsilon_w=0.01, epsilon=0.001):
    '''
    Factorize a matrix X into two non-negative matrices W and H using the Orthogonal Gradient Method (OGM) algorithm. 
    Algorithm based on the paper: doi: https://doi.org/10.1109/TSP.2012.2190406 
    X: the matrix to factorize
    Wt: the initial matrix W (randomly initialized)
    Ht: the initial matrix H (randomly initialized)
    E: matrix of ones with the same dimensions as Ht
    Sampled_mask: boolean matrix of the same size as X, with True values where we have sampled (observed) the matrix X. If you use the function sample_from_matrix, you can use the third output of the function
    beta: regularization parameter that determines the tradeoff between the sparsity of the solution and the accuracy of the factorization (higher beta means sparser solutions)
    max_iter: maximum number of iterations
    epsilon_h: convergence threshold for the H matrix
    epsilon_w: convergence threshold for the W matrix
    epsilon: convergence threshold for the algorithm
    '''
    iteration = 1
    prev_error = OGM_calculate_objective_function(Wt, Ht, X)

    grad_norm_H_list = []
    grad_norm_W_list = []
    obj_value_list = []
    sparsity_H_list = []

    while iteration < max_iter:
        H_new, grad_norm_H = OGM_H(Wt, Ht, X, E, beta, max_iter, epsilon_h)
        W_new, grad_norm_W = OGM_W(Wt, H_new, X, max_iter, epsilon_w)

        observed_error = OGM_calculate_objective_function_observed(W_new, H_new, X, Sampled_mask)
        unobserved_error = OGM_calculate_objective_function_unobserved(W_new, H_new, X, Sampled_mask)
        overall_error = OGM_calculate_objective_function(W_new, H_new, X)

        obj_value_list.append((observed_error, unobserved_error, overall_error))

        current_error = overall_error
        print(current_error)
        if prev_error - current_error <= epsilon:
            break

        prev_error = current_error
        print(iteration)

        Wt = W_new
        Ht = H_new

        grad_norm_H_list.append(grad_norm_H[-1])
        grad_norm_W_list.append(grad_norm_W[-1])
        sparsity_H_list.append(calculate_sparsity(H_new))

        iteration += 1

    return Wt, Ht, grad_norm_H_list, grad_norm_W_list, obj_value_list, sparsity_H_list