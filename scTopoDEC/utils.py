import os
import random
import numpy as np
import scanpy as sc
import keras
import tensorflow as tf
from keras import ops
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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


def estimate_optimal_noise(adata, 
                           n_clusters_guess=0, 
                           resolution=0.5,        
                           noise_min=0.1, 
                           noise_max=1.0, 
                           sil_min=0.0, 
                           sil_max=0.5):
    """
    Dynamically estimates the optimal noise_sd for scTopoDEC based on the 
    baseline separability (Silhouette Score) of the real-world dataset.
    
    Parameters:
    -----------
    adata : AnnData
        The raw or normalized single-cell dataset.
    n_clusters_guess : int
        A rough guess for baseline K-means to calculate separability. 
        If n_clusters_guess is None or 0, it auto-detects the clusters 
        using the Leiden algorithm.
    noise_min : float
        The minimum noise_sd to apply when clusters are highly distinct.
    noise_max : float
        The maximum noise_sd to apply when clusters overlap heavily.
    sil_min : float
        The theoretical minimum silhouette score expected (highly overlapping).
    sil_max : float
        The theoretical maximum silhouette score expected (highly distinct).
        
    Returns:
    --------
    float : The dynamically calculated noise_sd.
    """
    print("Calculating baseline manifold separability...")
    adata_tmp = adata.copy()
    
    # 1. Standard baseline reduction
    if 'X_pca' not in adata_tmp.obsm:
        # Check if data is already scaled/normalized. 
        # If the minimum value is less than 0, it has already been scaled.
        min_val = adata_tmp.X.min() 
        
        if min_val >= 0:
            print("Running internal preprocessing for noise estimation...")
            sc.pp.normalize_total(adata_tmp, target_sum=1e4)
            sc.pp.log1p(adata_tmp)
            if adata_tmp.n_vars > 2000:
                sc.pp.highly_variable_genes(adata_tmp, n_top_genes=2000)
                adata_tmp = adata_tmp[:, adata_tmp.var.highly_variable]
            sc.pp.scale(adata_tmp, max_value=10)
        else:
            print("Data appears already scaled. Skipping redundant preprocessing...")

        # Run PCA on the prepared data
        sc.tl.pca(adata_tmp, svd_solver='arpack')
    
    pca_features = adata_tmp.obsm['X_pca'][:, :50]
    
    # 2. Auto-detect or use fixed K
    if n_clusters_guess is None or n_clusters_guess <= 0:
        print("n_clusters not provided. Auto-detecting via Leiden...")
        # Leiden requires a neighborhood graph
        sc.pp.neighbors(adata_tmp, n_pcs=50)
        sc.tl.leiden(adata_tmp, resolution=resolution)
        
        labels = adata_tmp.obs['leiden'].values
        n_detected = len(np.unique(labels))
        print(f"Leiden found {n_detected} clusters at resolution {resolution}.")
        
        # Silhouette requires at least 2 clusters. If only 1 is found, 
        # the data is a single massive blob (highly overlapping).
        if n_detected < 2:
            print("Warning: Only 1 cluster detected. Returning maximum noise.")
            return float(noise_max)
    else:
        print(f"Using fixed KMeans with K={n_clusters_guess}...")
        kmeans = KMeans(n_clusters=n_clusters_guess, n_init=10, random_state=42)
        labels = kmeans.fit_predict(pca_features)
    
    # 3. Calculate Silhouette Score
    S = silhouette_score(pca_features, labels)
    print(f"Baseline Silhouette Score (S): {S:.4f}")
    
    # 4. Apply Bounded Linear Interpolation
    S_clamped = max(min(S, sil_max), sil_min)
    noise_sd = noise_max - ((S_clamped - sil_min) / (sil_max - sil_min)) * (noise_max - noise_min)
    
    print(f"Mapped to optimal noise_sd: {noise_sd:.4f}")
    return round(float(noise_sd), 4)


def compute_target_distribution(q, gamma=1.0):
    """
    Compute the target distribution p from soft labels q.
    p is designed to sharpen the clusters and emphasize high-confidence assignments.

    A gamma parameter (default 1.0) applies frequency smoothing to the denominator,
    preventing the mathematical collapse of small clusters during DEC optimization.
    """
    q = np.clip(q, 1e-10, 1.0)
    f_j = ops.sum(q, axis=0) + 1e-10
    weight = ops.power(q, 2) / ops.power(f_j, gamma)  
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
    input_mode   : 'raw', 'pca', 'tsne', 'umap', 'pca_dist', 'tsne_dist', 'umap_dist', 'knn', 'eff_res', or 'diffusion'
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
            activated_inner_product = tf.nn.tanh(inner_product)
            # dist_matrix = 1.0 - activated_inner_product
            dist_matrix = tg.get_latent_geometry(activated_inner_product, k=None)
            return dist_matrix
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
            data = data_compression(data, pca=True, knn=False, tsne=False, umap=False, n_components=n_components, k=k)
            return data.obsm['X_pca']
        elif input_mode == 'tsne':
            # Returns the tsNE coordinates
            data = data_compression(data, pca=True, knn=True, tsne=True, umap=False, n_components=n_components, k=k)
            return data.obsm['X_tsne']
        elif input_mode == 'umap':
            # Returns the UMAP coordinates
            data = data_compression(data, pca=True, knn=True, tsne=False, umap=True, n_components=n_components, k=k)
            return data.obsm['X_umap']
        elif input_mode == 'pca_dist':
            # Returns the PCA-based distance
            data = data_compression(data, pca=True, knn=False, tsne=False, umap=False, n_components=n_components, k=k)
            return squareform(pdist(data.obsm['X_pca']))
        elif input_mode == 'tsne_dist':
            # Returns the tSNE-based distance
            data = data_compression(data, pca=True, knn=True, tsne=True, umap=False, n_components=n_components, k=k)
            return squareform(pdist(data.obsm['X_tsne']))
        elif input_mode == 'umap_dist':
            # Returns the UMAP-based distance
            data = data_compression(data, pca=True, knn=True, tsne=False, umap=True, n_components=n_components, k=k)
            return squareform(pdist(data.obsm['X_umap']))
        elif input_mode == "knn":
            # Binary adjacency from Scanpy connectivities
            data = data_compression(data, pca=True, knn=True, tsne=False, umap=False, n_components=n_components, k=k)
            return (data.obsp['connectivities'].toarray() > 0).astype(np.float32)
        elif input_mode in ["eff_res", "diffusion"]:
            # Pre-computed Topological Distance
            data = data_compression(data, pca=True, knn=True, tsne=False, umap=False, n_components=n_components, k=k)
            knn_matrix = data.obsp['connectivities']
            return tg.get_dist(knn_matrix, distance=input_mode, t=t)

    raise ValueError(f"Unknown mode combination: Input={input_mode}, Latent={latent_mode}")


