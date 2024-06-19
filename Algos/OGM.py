import numpy as np
from copy import deepcopy

def frobenius_norm(A):
    return np.linalg.norm(A, 'fro')

def positive_project(H):
    return np.maximum(H, 0)

#gradient H
def compute_gradient_H(Wt, Y, X, E, beta):
    gradient = np.dot(np.dot(Wt.T, Wt), Y) - np.dot(Wt.T, X) + np.dot(beta, E)
    return gradient

#gradient W
def compute_gradient_W(Ht, Y, X):
    gradient = np.dot(np.dot(Y, Ht), Ht.T) - np.dot(X,Ht.T)
    return gradient

#Update algorithm H
def OGM_H(Wt, Ht, X, E, beta, max_iter=100, epsilon_h=0.01):
    Y = deepcopy(Ht)
    alpha = 1
    L = np.linalg.norm(np.dot(Wt.T, Wt), 2)
    k = 0
    H_prev = deepcopy(Ht)

    grad_norm_list_H = []

    while k < max_iter:
        grad_F = compute_gradient_H(Wt, Y, X, E, beta)
        H = positive_project(Y - (1/L) * grad_F)

        grad_norm = frobenius_norm(positive_project(grad_F))
        grad_norm_list_H.append(grad_norm)

        if grad_norm <= epsilon_h:
            break

        alpha_next = (1 + np.sqrt(4 * alpha**2 + 1)) / 2

        if k > 0:
            Y = H + ((alpha - 1) / alpha_next) * (H - H_prev)
        else:
            Y = H

        alpha = alpha_next
        H_prev = deepcopy(H)
        k += 1



    return H, grad_norm_list_H

#Update algorithm W
def OGM_W(Wt, Ht, X, max_iter=100, epsilon_w=0.01):
    Y = deepcopy(Wt)
    alpha = 1
    L = np.linalg.norm(np.dot(Ht, Ht.T), 2)
    k = 0
    W_prev = deepcopy(Wt)

    grad_norm_list_W = []

    while k < max_iter:
        grad_F = compute_gradient_W(Ht, Y, X)
        W = positive_project(Y - (1/L) * grad_F)

        grad_norm = frobenius_norm(positive_project(grad_F))
        grad_norm_list_W.append(grad_norm)

        if grad_norm <= epsilon_w:
            break

        alpha_next = (1 + np.sqrt(4 * alpha**2 + 1)) / 2

        if k > 0:
            Y = W + ((alpha - 1) / alpha_next) * (W - W_prev)
        else:
            Y = W

        alpha = alpha_next
        W_prev = deepcopy(W)
        k += 1


    return W, grad_norm_list_W

# #Initialize parameters!!
# n_features = data.shape[0]
# n_samples = data.shape[1]
# n_components = 50

# X = data
# Wt = np.random.rand(n_features, n_components)
# Ht = np.random.rand(n_components, n_samples)
# E = np.ones((n_components, n_samples))

# t = 1
# max_iter = 20
# epsilon_h = 0.01
# epsilon_w = 0.01
# epsilon = 0.001
# while t < max_iter:
#     #Update H
#     H, grad_norm_H = OGM_H(Wt, Ht, X_observed, E, beta=8)
#     H_new = H

#     #Update W
#     W, grad_norm_W = OGM_W(Wt, H_new, X_observed)
#     W_new = W

#     #Compute errors
#     observed_error = calculate_objective_function_observed(W_new, H_new, X, Omega)
#     unobserved_error = calculate_objective_function_unobserved(W_new, H_new, X, Omega)
#     overall_error = calculate_objective_function(W_new, H_new, X)

#     obj_value_list.append((observed_error, unobserved_error, overall_error))

#     current_error = overall_error
#     print(current_error)
#     #Convergence criteria
#     if prev_error - current_error <= epsilon:
#       break