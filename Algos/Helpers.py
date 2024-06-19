import numpy as np

def svd_threshold(A, threshold):
    """
    Apply the thresholding operator to the singular values of A
    A: matrix to apply the thresholding operator
    threshold: threshold value to apply to the singular values
    Returns reconstructed matrix A with singular values thresholded
    """
    U, s, V = np.linalg.svd(A, full_matrices=False)
    s = np.maximum(s - threshold, 0)
    return np.dot(U, np.dot(np.diag(s), V))

def sample_from_matrix(Matrix_to_sample, Ratio_to_keep, seed=42):
    '''
    Samples frpm a matrix A, keeping a ratio_to_keep of the values
    A: matrix to sample from
    ratio_to_keep: ratio of values to keep in the sampled matrix
    seed: seed for the random number generator, for ease of reproducibility
    Returns the sampled matrix X, the coordinates of the sampled values, and a boolean matrix of the same size as A, with True values where we have sampled the matrix A
    '''
    # initialize the random number generator with the seed
    rng = np.random.default_rng(seed)
    
    # calculate the number of samples to keep
    num_samples = int(Ratio_to_keep * Matrix_to_sample.size)
    
    sampled_indices = rng.choice(np.arange(Matrix_to_sample.size), size=num_samples, replace=False) # sample the indices of the values to keep
    sampled_mask = np.zeros(Matrix_to_sample.shape, dtype=bool) # create a boolean mask of the same size as the matrix
    sampled_mask.flat[sampled_indices] = True # set the values to True where we have sampled the matrix (true if we have sampled the value)
    
    sampled_matrix = np.zeros(Matrix_to_sample.shape) # create a matrix of zeros of the same size as the matrix to sample
    sampled_matrix.flat[sampled_indices] = Matrix_to_sample.flat[sampled_indices] # set the values of the sampled matrix to the values of the original matrix
        
    
    return sampled_matrix, sampled_indices, sampled_mask

def normalize_rows(X):
    '''
    Normalize the rows of a matrix X, by dividing each row by the maximum value of the row (max normalization)
    '''
    row_max = X.max(axis=1)
    X = X / row_max[:, np.newaxis]
    return X

def normalize_columns(X):
    '''
    Normalize the columns of a matrix X, by dividing each column by the maximum value of the column (max normalization)
    '''
    col_max = X.max(axis=0)
    X = X / col_max[np.newaxis, :]
    return X

def insampling_error_relative(X_sampled, X_reconstructed, Sampled_mask, norm='fro'):
    """
    Compute the relative insampling error between the original matrix X_sampled and the reconstructed matrix X_reconstructed
    X_sampled: the sampled matrix with zeros where we have not sampled
    X_reconstructed: reconstructed/estimated matrix we want to compare to X_sampled
    Sampled_mask: boolean matrix of the same size as X_sampled, with True values where we have sampled the matrix X_sampled. If you use the function sample_from_matrix, you can use the third output of the function
    """
    
    X_reconstructed_sampled = np.where(Sampled_mask, X_reconstructed, 0) # sample the reconstructed matrix
    error = np.linalg.norm(X_sampled - X_reconstructed_sampled, ord=norm)
    relative_error = error / np.linalg.norm(X_sampled, ord=norm)
    
    return relative_error
    
def out_of_sample_error_relative(X_original, X_reconstructed, Sampled_mask, norm='fro'):
    """
    Compute the relative out-of-sample error between the original matrix X_original and the reconstructed matrix X_reconstructed divided by the complement of the sampled matrix
    X_original: the original matrix
    X_reconstructed: reconstructed/estimated matrix we want to compare to X_original
    Sampled_mask: boolean matrix of the same size as X_original, with True values where we have sampled the matrix X_original. If you use the function sample_from_matrix, you can use the third output of the function
    """
    X_original_complement = np.where(~Sampled_mask, X_original, 0)
    X_reconstructed_complement = np.where(~Sampled_mask, X_reconstructed, 0)
    error = np.linalg.norm(X_original_complement - X_reconstructed_complement, ord=norm)
    relative_error = error / np.linalg.norm(X_original_complement, ord=norm)
    return relative_error

def general_error_relative(X_original, X_reconstructed, norm='fro'):
    """
    Compute the relative error between the original matrix X_original and the reconstructed matrix X_reconstructed
    X_original: the original matrix
    X_reconstructed: reconstructed/estimated matrix we want to compare to X_original
    """
    error = np.linalg.norm(X_original - X_reconstructed, ord=norm)
    relative_error = error / np.linalg.norm(X_original, ord=norm)
    
    return relative_error

def euclideanDivergence(x,y):
    return .5*(x-y)**2

def KLDivergence(x,y):
    # the Kullback-Leibler divergence function
    return x * np.log(x/y)-x+y

def ISDivergence(x,y):
    # the Itakura-Saito divergence function
    return x/y - np.log(x/y)-1

def positive_projection(F):
    # for loc in np.argwhere(F<0):
    #     F[loc[0],loc[1]] = 0
    return np.maximum(F, 0)

def calculate_sparsity(X):
    '''
    Calculate the sparsity of a matrix X, defined as the percentage of zero elements in the matrix
    X: the matrix to calculate the sparsity of
    '''
    total_elements = X.size
    zero_elements = np.count_nonzero(X == 0)
    sparsity = (zero_elements / total_elements) * 100
    return sparsity

def generate_simulated_IMS_matrix(m=50000, n=500, rank=100, seed=42, random_mean=1.0, random_scale=1, batch_contrast=2, batch_abondance=200, remove_batches=True):
    """
    DEPRECATED: DON'T USE
    m: number of rows
    n: number of columns
    rank: rank of the matrix, constructed from 2 matrices W and H of size m x rank and rank x n
    seed: seed for the random number generator, for ease of reproducibility
    random_mean: mean of the gamma distribution for the random values
    random_scale: scale of the gamma distribution for the random values
    batch_contrast: contrast of the batch values, during generation, 'batches' are added to the W and H matrices, to simulate the original data
    batch_abondance: number of batches to add to the W and H matrices
    remove: if True, remove some values in the batches, to simulate the original data
    
    Run the function to generate a simulated IMS matrix, with the given parameters
    The function makes first two matrices W and H of size m x rank and rank x n, and then computes the product W*H, which is the simulated IMS matrix
    Run the function with no input to get the default values
    """
    
    rng=np.random.default_rng(seed)
    W_rows, W_cols = m, rank
    H_rows, H_cols = rank, n
    
    W = rng.gamma(shape=random_mean, scale=random_scale, size=(W_rows, W_cols))
    H = rng.gamma(shape=random_mean, scale=random_scale/100, size=(H_rows, H_cols))
    
    def add_batches(A,num_batches, batch_size, high_value, remove=True):
        for _ in range(num_batches):
            row_start = np.random.randint(0, A.shape[0] - batch_size[0])
            col_start = np.random.randint(0, A.shape[1] - batch_size[1])
            A[row_start:row_start + batch_size[0], col_start:col_start + batch_size[1]] += high_value+np.random.normal(0,high_value/2,batch_size)
        # Set some values to zero in the batches
        if remove:
            for _ in range(num_batches):
                row_start = np.random.randint(0, A.shape[0] - batch_size[0])
                col_start = np.random.randint(0, A.shape[1] - batch_size[1])
                A[row_start:row_start + batch_size[0], col_start:col_start + batch_size[1]] = 1e-2
        A = np.clip(A, 1e-2, None)
        return A
    
    W = add_batches(W, num_batches=batch_abondance, batch_size=(np.max([int(m/5),1]), 2), high_value=batch_contrast,remove=remove_batches)
    H = add_batches(H, num_batches=batch_abondance*5, batch_size=(2,np.max([int(n/25),1])), high_value=batch_contrast/40, remove=remove_batches)
    simulated_X = np.dot(W,H)
    simulated_X = normalize_columns(simulated_X)
    return simulated_X