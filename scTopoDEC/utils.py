import os
import random
import numpy as np
import keras
import tensorflow as tf
from keras import ops
from scipy.spatial.distance import pdist, squareform
from . import topogeom as tg
from .io import data_compression



def set_reproducibility(seed=0):
    # Environment and Python seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    
    # Math and array seeds
    np.random.seed(seed)
    
    # Deep learning framework seeds
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def compute_target_distribution(q):
    """
    Compute the target distribution p from soft labels q.
    p is designed to sharpen the clusters and emphasize high-confidence assignments.
    """
    q = np.clip(q, 1e-10, 1.0)
    weight = ops.power(q, 2) / (ops.sum(q, axis=0) + 1e-10) # Add 1e-8 to avoid division by zero 
    p = weight / (ops.sum(weight, axis=1, keepdims=True) + 1e-10)
    return np.nan_to_num(p)


def density_scale(t, select_size, indices=None):
    """
    Compute a sample of pairwise distance to find the scale.
    This function is used to make two different spaces comparable when
    calculate topological loss
    """
    # Randomly subset of points to save memory
    n_points = tf.shape(t)[0]
    sample_size = tf.minimum(n_points, select_size)

    if indices is None:
        indices = tf.random.shuffle(tf.range(n_points))[:sample_size]

    # Check if input is a square distance matrix [N, N] or coordinates [N, D]
    is_distance_matrix = (tf.rank(t) == 2 and tf.shape(t)[0] == tf.shape(t)[1])

    if is_distance_matrix:
        # Sub-matrix sampling: gather rows then gather columns
        sample = tf.gather(tf.gather(t, indices, axis=0), indices, axis=1)
        D = sample
    else:
        # Coordinate sampling
        sample = tf.gather(t, indices)
        # Compute pairwise Euclidean distances: ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        r = tf.reduce_sum(sample*sample, 1, keepdims=True)
        sq_dist = r - 2*tf.matmul(sample, tf.transpose(sample)) + tf.transpose(r)
        D = tf.sqrt(tf.maximum(sq_dist, 1e-12))

    # Calculate scale factor (average distance)
    avg_dist = tf.reduce_mean(D) + 1e-8
    t_scaled = t / avg_dist

    return t_scaled, indices


def get_topo_representation(data, input_mode='pca', latent_mode='raw', n_components=30, k=15, t=8, 
                            is_latent=False):
    """
    Standardizes inputs for Persistent Homology.
    
    # Arguments
    data         : adata (for input space) or Tensor (for latent space). 
    input_mode   : 'raw', 'pca', 'umap', 'pca_dist', 'umap_dist', 'knn', 'eff_res', or 'diffusion'
    latent_mode  : 'raw', 'inner_product','euclid_dist', 'knn', 'eff_res', or 'diffusion'
    n_components : Number of components for PCA and UMAP
    k            : Number of neighbors for graph modes
    t            : Diffusion time (steps)
    is_latent    : If True, uses differentiable TensorFlow logic

    # Return
        Preprocessed representation
    """
    
    # --- TRACK 1: LATENT SPACE (Dynamic/TensorFlow) ---
    if is_latent:
        # data is a Tensor 'z' (batch_size, latent_dim)
        if latent_mode == 'raw':
            return data
        elif latent_mode == 'inner_product':
            inner_product = tf.matmul(data, data, transpose_b=True)
            return inner_product
        elif latent_mode == 'euclid_dist':
            # Full (batch, batch) Euclidean distance
            return tg.get_latent_geometry(data, k=None)
        elif latent_mode == 'knn':
            # Symmetric 0/1 Adjacency matrix
            knn_matrix = tg.get_latent_geometry(data, k=k, binary_nearest=True)
            return tf.cast(knn_matrix, tf.float32)
        elif latent_mode in ["eff_res", "diffusion"]:
            # Warning: Running these in train_step is computationally expensive
            knn_matrix = tg.get_latent_geometry(data, k=k, binary_nearest=True)
            dist_matrix = tg.get_dist(knn_matrix, distance=latent_mode, t=t)
            return tf.cast(dist_matrix, tf.float32)
            
    # --- TRACK 2: INPUT SPACE (Static/Scanpy/NumPy) ---
    else:
        # data is the 'adata' object
        if input_mode == 'raw':
            X = data.X.toarray() if hasattr(data.X, "toarray") else data.X
            return X
        elif input_mode == 'pca':
            # Returns the PCA coordinates
            data = data_compression(data, pca=True, knn=False, umap=False, n_components=n_components, k=k)
            return data.obsm['X_pca']
        elif input_mode == 'umap':
            # Returns the UMAP coordinates
            data = data_compression(data, pca=True, knn=True, umap=True, n_components=n_components, k=k)
            return data.obsm['X_umap']
        elif input_mode == 'pca_dist':
            # Returns the PCA-based distance
            data = data_compression(data, pca=True, knn=False, umap=False, n_components=n_components, k=k)
            return squareform(pdist(data.obsm['X_pca']))
        elif input_mode == 'umap_dist':
            # Returns the UMAP-based distance
            data = data_compression(data, pca=True, knn=True, umap=True, n_components=n_components, k=k)
            return squareform(pdist(data.obsm['X_umap']))
        elif input_mode == "knn":
            # Binary adjacency from Scanpy connectivities
            data = data_compression(data, pca=True, knn=True, umap=False, n_components=n_components, k=k)
            return (data.obsp['connectivities'].toarray() > 0).astype(np.float32)
        elif input_mode in ["eff_res", "diffusion"]:
            # Pre-computed Topological Distance
            data = data_compression(data, pca=True, knn=True, umap=False, n_components=n_components, k=k)
            knn_matrix = data.obsp['connectivities']
            return tg.get_dist(knn_matrix, distance=input_mode, t=t)

    raise ValueError(f"Unknown mode combination: Input={input_mode}, Latent={latent_mode}")


