import numpy as np
from numpy.linalg import inv, norm
from Algos.Helpers import positive_projection, insampling_error_relative, projection_mask_operator

# random_m = 500
# random_n = 500
# random_rank = 20
# L_matrix = np.random.rand(random_m,random_rank)
# R_matrix = np.random.rand(random_rank,random_n)
# print(L_matrix.shape)
# print(R_matrix.shape)
# D_matrix = np.diag(np.arange(1,random_rank+1))
# print(D_matrix.shape)
# random_data = L_matrix@D_matrix@R_matrix
# sampled_random_data75,coords75,coords75TF = sample_from_matrix(random_data,0.75)
# sampled_random_data50,coords50,coords50TF = sample_from_matrix(random_data,0.5)
# sampled_random_data25,coords25,coords25TF = sample_from_matrix(random_data,0.25)


def NMFC_sparsity(M, k, Sampled_mask, alpha_factor, alpha_choice, max_iter=500, tol=1e-5):

    
    # Define a projection operator which may involve some form of dimensionality reduction or approximation
    A = projection_mask_operator(M, M)
    m, n = A.shape  # Dimensions of the matrix A

    # Calculate the Frobenius norm of matrix A
    frobenius_A = norm(A, 'fro')

    if alpha_choice == 1:
        alpha = alpha_factor * frobenius_A * max(m, n) / k
        beta = (n * alpha) / m
        rho = beta
        lambda_param = 0.3

    elif alpha_choice == 2:
        alpha = alpha_factor
        beta = alpha
        rho = beta
        lambda_param = 0.3

    # Alpha controls the trade-off between the approximation accuracy of matrix X in reconstructing matrix A and the regularization applied to X,
    # scaling the impact based on the matrix dimensions and the Frobenius norm of A.
    print(f'alpha: {alpha}   beta: {beta}')
    #  Beta adjusts the regularization strength on matrix Y in proportion to alpha,
    # modified by the relative dimensions of A, ensuring balanced regularization across both factors in the decomposition.
    gamma = 1.618 # Gamma modulates the update speed of the dual variables in the ADMM algorithm, influencing the convergence rate and stability of the optimization process.

    # Initialize matrices X, Y, Z, U, V, Lambda, and Pi
    X = np.random.rand(m, k)
    # AKA W matrix

    Y = np.random.rand(k, n)
    # AKA H matrix

    Z = A.copy()  # Start with Z being equal to A
    
    Omega = np.zeros(Y.shape)
    S = np.zeros(Y.shape)
    E = np.ones(Y.shape)

    U = np.zeros(X.shape)
    V = np.zeros(Y.shape)

    # lagrange multipliers
    Lambda = np.zeros(X.shape)
    # Lambda thus captures the gap between X and U over iterations, applying corrections based on this discrepancy.
    Pi = np.zeros(Y.shape)
    # The adjustments in Pi help correct and guide the updates of Y to align more closely with V across iterations.
    intermediate_error = []
    
    # Main ADMM iteration loop
    for i in range(max_iter):
        # Update X by solving the X subproblem, includes matrix inversion and multiplication
        X_next = positive_projection((Z @ Y.T + alpha * U - Lambda) @ inv(Y @ Y.T + alpha * np.eye(k)))

        # Update Y by solving the Y subproblem
        Y_next = positive_projection(inv(X_next.T @ X_next + beta * np.eye(k) + rho * np.eye(k)) @ (X_next.T @ Z + beta * V - Pi - Omega + rho * S))

        # Update Z by projecting onto some subspace defined by projection_operator
        Z_next = X_next @ Y_next + projection_mask_operator(M - (X_next @ Y_next), M)

        # Project U and V to ensure they remain positive
        U_next = positive_projection(X_next + Lambda / alpha)
        V_next = positive_projection(Y_next + Pi / beta)
        S_next = Omega / rho + Y_next - lambda_param * E

        # Update dual variables Lambda and Pi
        Lambda_next = Lambda + gamma * alpha * (X_next - U_next)
        Pi_next = Pi + gamma * beta * (Y_next - V_next)
        Omega_next = Omega + gamma*rho*(Y_next - S_next)

        f_k = norm(projection_mask_operator(X @ Y - A, M), 'fro') / frobenius_A
        f_k1 = norm(projection_mask_operator(X_next @ Y_next - A, M), 'fro') / frobenius_A
        
        intermediate_error.append(insampling_error_relative(M, X_next @ Y_next, Sampled_mask))
        print(i, insampling_error_relative(M, X_next @ Y_next, Sampled_mask))
        
        if np.abs(f_k1 - f_k) / np.maximum(1, np.abs(f_k)) <= tol:
            print('first', i)
            return X_next, Y_next
            # norm(M - X_next @ Y_next, 'fro') / norm(M, 'fro')
        # Convergence check based on the relative change in Frobenius norm of the projection operator
        elif f_k <= tol:
            print('second', i)
            return X_next, Y_next

        # Update variables for the next iteration
        X, Y, Z, U, V, Lambda, Pi, Omega, S = X_next, Y_next, Z_next, U_next, V_next, Lambda_next, Pi_next, Omega_next, S_next
    print('here')  # Indicates end of iteration if convergence criterion is not met

    return X, Y
