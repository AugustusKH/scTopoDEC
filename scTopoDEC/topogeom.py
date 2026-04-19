import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import scipy.sparse as sp
import scipy.linalg
import tensorflow as tf
from keras import ops


###### These functions below are used to convert a latent space to an input for persistent homology ######

def build_latent_adjacency(knn_distances, knn_indices):
    """
    Constructs a latent adjacency matrix from kNN results.
    
    # Arguments
    knn_distances : Tensor of shape (batch_size, k)
    knn_indices : Tensor of shape (batch_size, k)
    batch_size : Number of cells in the current batch
    
    # Return
        A dense adjacency matrix of shape (batch_size, batch_size)
    """
    # Create a range of row indices: [0, 0, ..., 1, 1, ..., N, N]
    # Each row index must be repeated k times to match the knn_indices
    batch_size = tf.shape(knn_distances)[0]
    row_indices = tf.range(batch_size, dtype=tf.int32)
    row_indices = tf.expand_dims(row_indices, 1) # Shape (batch_size, 1)
    row_indices = tf.tile(row_indices, [1, tf.shape(knn_indices)[1]]) # Shape (batch_size, k)

    # Flatten everything to create coordinate pairs (row, col)
    # This creates a list of coordinates for every 'edge' in our kNN graph
    indices = tf.stack([tf.reshape(row_indices, [-1]), 
                        tf.reshape(knn_indices, [-1])], axis=1)
    
    # Flatten the distances to be the values at those coordinates
    values = tf.reshape(knn_distances, [-1])

    # Scatter the values into a zero-filled NxN matrix
    # tf.scatter_nd places 'values' at the specified 'indices'
    adj = tf.scatter_nd(indices, values, shape=[batch_size, batch_size])
    adj_symmetric = 0.5 * (adj + tf.transpose(adj))
    
    return adj_symmetric 


def build_latent_adjacency_binary(knn_indices):
    """
    Constructs a symmetric binary (0/1) adjacency matrix.

    # Argument
    knn_indices : Tensor of shape (batch_size, k)

    # Return 
        A binary dense adjacency matrix of shape (batch_size, batch_size),
        where 1 if cell i is a neighbor of j OR j is a neighbor of i.
    """
    batch_size = tf.shape(knn_indices)[0]
    k = tf.shape(knn_indices)[1]

    # Create coordinate pairs (row, col)
    row_indices = tf.range(batch_size, dtype=tf.int32)
    row_indices = tf.reshape(row_indices, [-1, 1]) 
    row_indices = tf.tile(row_indices, [1, k]) 

    indices = tf.stack([tf.reshape(row_indices, [-1]), 
                        tf.reshape(knn_indices, [-1])], axis=1)
    
    # Scatter ones (Binary weights)
    ones = tf.ones([batch_size * k], dtype=tf.float32)
    
    # Create the initial directed adjacency matrix
    adj = tf.scatter_nd(indices, ones, shape=[batch_size, batch_size])
    
    # Symmetrize using Logical OR logic
    # If A->B is 1 OR B->A is 1, the symmetric edge is 1.
    adj_symmetric = adj + tf.transpose(adj)
    
    # Use tf.cast to turn any values > 0 into 1.0
    adj_binary = tf.where(adj_symmetric > 0, 1.0, 0.0)
    
    # Remove self-loops 
    adj_binary = tf.linalg.set_diag(adj_binary, tf.zeros([batch_size]))
    
    return adj_binary


def get_latent_geometry(z, k=None, binary_nearest=True):
    """
    Computes distances in the latent space Z.
    
    # Arguments
    z : Latent tensor of shape (batch_size, latent_dim)
    k : If None, returns full distance matrix. If int, returns kNN distances.
    
    # Return
        Distance matrix 
    """
    # Compute Full Pairwise Euclidean Distance Matrix (Vectorized)
    # Formula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    inner_product = tf.matmul(z, z, transpose_b=True)
    square_norm = tf.reduce_sum(tf.square(z), axis=1, keepdims=True)
    
    # Distance squared calculation
    D_sq = square_norm - 2 * inner_product + tf.transpose(square_norm)
    
    # Numerical stability: ensure no negative values before sqrt
    D = tf.sqrt(tf.maximum(D_sq, 1e-12))

    # Logic for kNN filtering
    if k is not None:
        # We use -D because top_k finds the largest values, 
        # and we want the smallest distances.
        knn_vals, knn_indices = tf.nn.top_k(-D, k=k)
        knn_distances = -knn_vals # Convert back to positive
        
        if binary_nearest:
            D = build_latent_adjacency_binary(knn_indices)
        else:
            D = build_latent_adjacency(knn_distances, knn_indices)
        return D
    else:
        return D


########### These codes below are used to calculate kNN distance ###########

def get_sknn(knn_matrix):
    """
    Takes a Scanpy adjacency matrix and returns a symmetric, unweighted graph.
    
    # Arguments
    knn_matrix : adata.obsp['connectivities'] or adata.obsp['distances']
    
    # Return 
        sparse coo adjacency matrix (symmetric and unweighted)
    """
    # Change to compressed sparse row (CSR) form
    sknn = knn_matrix.copy().tocsr()

    # Make the matrix unweighted (binary: 1 if neighbor, 0 otherwise)
    sknn.data = np.ones_like(sknn.data)

    # Symmetrize (force mutual neighbors)
    # A = max(A, A_transpose)
    sknn = sknn.maximum(sknn.transpose()).tocoo()

    return sknn


def compute_laplacian(A, normalization="none"):
    """
    Computes the Laplacian based ona an adjacency matrix given as np.ndarray or scipy.sparse matrix. 
    Based on code by Enrique Fita Sanmartin.
    
    # Arguments
    A : adjacency matrix given as np.ndarray or scipy.sparse matrix
    normalization : whether to use no normalization ("none"), random walk normalization ("rw") or symmetric 
    normalization ("sym")
    
    # Return 
        Laplacian matrix in the same format as A
    """
    # Determine if input is Sparse or Dense
    is_sparse = sp.issparse(A)
    
    # Compute degree matrix (D)
    degs = np.asarray(A.sum(axis=1)).flatten()
    
    if is_sparse:
        D = sp.diags(degs, format="csr")
    else:
        D = np.diag(degs)

    # Compute unnormalized Laplacian: L = D - A
    L = D - A

    # Handle normalization
    if normalization != "none":
        degs_safe = np.where(degs > 0, degs, 1e-12)

        if normalization == "rw": # Random Walk: L = D^-1 * L
            d_inv = degs_safe ** -1
            D_inv = sp.diags(d_inv) if is_sparse else np.diag(d_inv)
            L = D_inv @ L
            
        elif normalization == "sym": # Symmetric: L = D^-0.5 * L * D^-0.5
            d_inv_sqrt = degs_safe ** -0.5
            D_inv_sqrt = sp.diags(d_inv_sqrt) if is_sparse else np.diag(d_inv_sqrt)
            L = D_inv_sqrt @ L @ D_inv_sqrt
            
    return L


def compute_effective_resistance_connected(A):
    """
    Computes the effective resistance using the pseudoinverse of the Laplacian L^+ of a connected graph.
    Formula: EffR[i,j] = L+[i,i] + L+[j,j] - 2*L+[i,j]
    Based on code by Enrique Fita Sanmartin.
   
    # Arguments
    A : adjacency matrix (numpy or scipy.sparse array)
    
    # Return 
        All pairs of effective resistance distances (numpy array)
    """
    n = A.shape[0]
    
    # Get the Laplacian 
    L = compute_laplacian(A, normalization="none")
    
    # The "Pseudo-inverse" trick for connected graphs:
    # Adding 1/n to all entries makes the Laplacian invertible (shifting the zero eigenvalue)
    # However, 'np.ones' is dense. We only do this if N is small enough.
    if n > 5000:
        print(f"Warning: Matrix size {n}x{n} is large. Effective Resistance is memory intensive.")

    # Convert to dense only for the inversion step
    if sp.issparse(L):
        L_dense = L.toarray()
    else:
        L_dense = L

    # Mathematically: L_pinv = (L + J/n)^-1 - J/n, where J is all-ones matrix
    # This is the most stable way to get the Moore-Penrose pseudo-inverse for a Laplacian
    J_n = np.ones((n, n)) / n
    L_pinv = np.linalg.inv(L_dense + J_n) - J_n

    # Extract the diagonal (L+[i,i])
    # Linv_diag is a vector of shape (n, 1)
    Linv_diag = np.diag(L_pinv).reshape((n, 1))

    # Broadcast to compute all-pairs: EffR[i,j] = diag_i + diag_j - 2*Lpinv_ij
    # This uses numpy broadcasting which is much faster than loops
    EffR = Linv_diag + Linv_diag.T - 2 * L_pinv

    # Clean up small numerical errors (resistances can't be negative)
    EffR = np.maximum(EffR, 0)
    
    return EffR


def compute_effective_resistance(A, disconnect=True):
    """
    Computes the effective resistance using the pseudoinverse of the Laplacian L^+ of an arbitrary graph. We will compute
    the effective resistance on each component separately and set the resistance between different components to inf.
    
    # Arguments
    A : Adjacency matrix (np.ndarry or scipy.sparce matrix)
    disconnect : whether to compute the effective resistance for each connected component separately
   
    # Rreturn 
        All pairs of effective resistance distances (np.ndarray)
    """
    n = A.shape[0]
    
    if disconnect:
        # Identify connected components
        # component_labels assigns each cell an ID (e.g., [0, 0, 1, 0, 1])
        n_components, component_labels = sp.csgraph.connected_components(A)
        
        # Initialize with infinity
        # If two cells are on different islands, resistance is technically infinite
        EffR = np.full((n, n), np.inf)

        if sp.issparse(A):
            A = A.tocsr() # CSR is required for fast slicing of components

        for i in range(n_components):
            # Identify which cells belong to the current island
            component_indices = np.where(component_labels == i)[0]
            
            # If the island is just one cell, resistance is 0 (already handled by diag)
            if len(component_indices) > 1:
                # Slice the adjacency matrix to isolate this island
                A_sub = A[component_indices, :][:, component_indices]
                
                # Compute resistance for this sub-graph
                EffR_sub = compute_effective_resistance_connected(A_sub)
                
                # Map the local resistances back to the global NxN matrix
                # ix_ spreads the results into the correct row/column slots
                EffR[np.ix_(component_indices, component_indices)] = EffR_sub

    else:
        # If we assume the graph is one piece, use the direct method
        EffR = compute_effective_resistance_connected(A)

    # Handle the infinity problem for TDA
    # Persistent homology cannot handle 'inf'. We replace it with a 
    # value large enough to ensure these components remain separate 
    # in the persistence diagram.
    finite_mask = np.isfinite(EffR)
    if np.any(finite_mask):
        max_val = np.max(EffR[finite_mask])
        EffR[np.isinf(EffR)] = max_val * 2
    else:
        # Fallback if no finite values exist
        EffR[np.isinf(EffR)] = 10.0

    return EffR


def correct_eff_res(d, adj):
    """
    Corrects the effective resistance distance matrix for large n (à la von Luxburg).
    Formula: d_corr = d - (1/deg_i + 1/deg_j) + 2*adj_ij/(deg_i * deg_j)
    
    # Arguments
    d : effective resistance distance matrix
    adj : adjacency matrix as coo matrix
    
    # Return 
        corrected effective resistance distance matrix
    """
    # Compute degrees 
    degs = np.asarray(adj.sum(axis=1)).flatten()
    degs_safe = np.where(degs > 0, degs, 1e-12)
    
    # Compute the degree distance term: (1/deg_i + 1/deg_j)
    # inv_degs is (n, 1), inv_degs.T is (1, n)
    inv_degs = (1.0 / degs_safe).reshape(-1, 1)
    deg_dist = inv_degs + inv_degs.T
    
    # Distance to self is always 0
    np.fill_diagonal(deg_dist, 0)
    
    # Compute the correction Term: 2 * adj / (deg_i * deg_j)
    if sp.issparse(adj):
        # Use sparse math: multiply each edge (i,j) by 2/(deg_i * deg_j)
        adj_corr = adj.tocoo().copy()
        # row and col are the indices of existing connections
        rows, cols = adj_corr.row, adj_corr.col
        adj_corr.data = 2.0 * adj_corr.data / (degs_safe[rows] * degs_safe[cols])
        # Convert back to dense only at the final addition step
        adj_term = adj_corr.toarray()
    else:
        # If already dense
        adj_term = 2.0 * adj / (degs_safe.reshape(-1, 1) * degs_safe.reshape(1, -1))

    # 4. Final Correction
    d_corrected = d - deg_dist + adj_term
    
    # Numerical stability: distances shouldn't be negative
    return np.maximum(d_corrected, 0)


def get_eff_res_dist(knn_matrix, corrected=True, disconnect=True):
    """
    Computes effective resistence distance on sknn graph
    
    # Arguments
    knn_matrix : adata.obsp['connectivities'] or adata.obsp['distances']
    corrected : whether to do the von Luxburg correction
    disconnect : whether to compute the effective resistance on each connected component separately
    
    # Return
        Pairwise distance matrix
    """

    # Compute symmetric kNN graph
    sknn_coo = get_sknn(knn_matrix)

    # Invert as edge weights are reciprocal of resistance
    sknn_coo.data = 1 / sknn_coo.data

    # Compute effective resistance
    d_eff = compute_effective_resistance(sknn_coo, disconnect=disconnect)

    # Optionally: correct via von Luxburg fix
    if corrected:
        d_eff = correct_eff_res(d_eff, sknn_coo)
    return d_eff


def get_diffusion_power(knn_matrix, t=8, include_self=True, return_D_inv=False):
    """
    Computes the diffusion matrix P^t using a pre-calculated kNN graph.
    
    # Arguments
    knn_matrix : adata.obsp['connectivities'] or adata.obsp['distances']
    t : Diffusion time (steps)
    include_self : Add self-loops for a 'lazy' random walk
    
    # Return 
        P_t
    """
    # Get an adjacency matrix
    A = get_sknn(knn_matrix).tocsr()

    # Add self-loops (Lazy Random Walk)
    # This prevents the walk from 'oscillating' and ensures convergence
    if include_self:
        A = A + sp.eye(A.shape[0], format="csr")

    # Compute row-normalization: P = D^-1 * A
    # A transition matrix must have rows that sum to 1.0 (probabilities)
    degrees = np.asarray(A.sum(axis=1)).flatten()
    degrees_safe = np.where(degrees > 0, degrees, 1e-12)
    
    D_inv = sp.diags(1.0 / degrees_safe)
    P = D_inv @ A # This is our 1-step transition matrix

    # Matrix power (diffusion step)
    # As we multiply P by itself t times, we find the probability of 
    # jumping from cell i to cell j in exactly t steps.
    P_dense = P.toarray()
    P_t = np.linalg.matrix_power(P_dense, t)

    if return_D_inv:
        return P_t, D_inv
    else:
        return P_t


def get_diffusion_dist(knn_matrix, t=8, include_self=True):
    """
    Computes diffusion distance on sknn graph
    
    # Arguments
    knn_matrix : Adjacency matrix (adata.obsp['connectivities'])
    t : diffusion time
    kernel : kernel to use, must be one of "sknn", "gaussian"
    
    # Return
        Pairwise distance matrix
    """
    # Get t-th power of the diffusion matrix
    P_t, D_inv = get_diffusion_power(
        knn_matrix, 
        t=t, 
        include_self=include_self, 
        return_D_inv=True
    )

    # Extract 1/degrees vector
    # D_inv is a sparse diagonal matrix; we need the actual values to scale P_t
    inv_degs = np.asarray(D_inv.diagonal()).flatten()
    
    # Apply the diffusion weighting
    # We scale each column of P_t by sqrt(1/degree_j)
    weighted_P_t = P_t * np.sqrt(inv_degs)

    # Compute pairwise euclidean distance in diffusion space
    # The distance between two cells is the L2 norm of their diffusion profiles
    dist_matrix = squareform(pdist(weighted_P_t))

    # Global scale correction
    # Multiplies by sqrt of the sum of original degrees (which is 1 / inv_degs)
    scale_factor = np.sqrt(np.sum(1.0 / inv_degs))
    
    return dist_matrix * scale_factor


def get_dist(knn_matrix, distance="eff_res", **kwargs):
    """
    Wrapper for all distances.
    
    # Arguments
    knn_matrix : Adjacency matrix (adata.obsp['connectivities'])
    distance : distance to compute, must be one of "eff_res", "diffusion"
    kwargs : key word arguments for the distance function
    
    # Return
        Distance matrix
    """
    if distance == "eff_res":
        kwargs.pop('t', None) # Remove 't' from kwargs if it exists before calling get_eff_res_dist
        dist = get_eff_res_dist(knn_matrix, **kwargs)
    elif distance == "diffusion":
        dist = get_diffusion_dist(knn_matrix, **kwargs)
    else:
        raise NotImplementedError(f"Distance {distance} not implemented")
    return dist