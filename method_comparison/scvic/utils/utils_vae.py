import logging
from typing import Tuple

import numpy as np
import scanpy as sc
from scvi.inference import Posterior
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


def Louvain(
        posterior: Posterior,
        n_neighbors: int = 15,
        resolution: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes cluster parameters by running Louvain community detection 
    on the latent space neighborhood graph.
    """
    # Unpack posterior mapping safely
    latent, _, _ = posterior.get_latent()
    n_samples = latent.shape[0]
    
    sc_latent = sc.AnnData(X=latent)
    if n_samples > 200000:
        # Fixed typo: "quit" -> "quite", "20,0000" -> "200,000"
        log_message = (
            "Number of samples is quite large, "
            "resample 200,000 samples to estimate parameters"
        )
        logger.info(log_message)
        # Explicitly copy to avoid modifying views or slicing locked arrays
        sc_latent = sc_latent[np.random.choice(n_samples, 200000, replace=False)].copy()

    log_message = "Construct KNN graph before Louvain in scanpy"
    logger.info(log_message)
    sc.pp.neighbors(sc_latent, n_neighbors=n_neighbors, use_rep="X")

    log_message = "Run Louvain"
    logger.info(log_message)
    sc.tl.louvain(sc_latent, resolution=resolution)
    
    louvain_labels = sc_latent.obs['louvain']
    louvain_labels = np.asarray(louvain_labels, dtype=int)
    n_clusters = np.unique(louvain_labels).shape[0]
    
    if n_clusters <= 1:
        # Fixed: Changed catastrophic exit() to an informative ValueError
        raise ValueError(
            f"Error: Only {n_clusters} cluster detected. The resolution: "
            f"{resolution} is too small, choose a larger resolution value."
        )
        
    ratio = []
    mu = []
    for i in range(n_clusters):
        indices = (louvain_labels == i)
        ratio.append(int(np.sum(indices)))
        
        # Calculate cluster centroid
        cluster_mean = latent[indices].mean(axis=0).reshape(1, -1)
        mu.append(cluster_mean)
        
    ratio = np.array(ratio, dtype=np.float32) / n_samples
    mu = np.concatenate(mu, axis=0)
    
    return mu, ratio, latent, louvain_labels


def GMM(
        posterior: Posterior,
        n_components: int,
        covariance_type: str = 'full'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initializes cluster parameters by fitting a parametric Gaussian Mixture Model
    directly to the biological latent embeddings.
    """
    latent, _, _ = posterior.get_latent()
    n_samples = latent.shape[0]
    
    if n_samples > 200000:
        log_message = (
            "Number of samples is quite large, "
            "resample 200,000 samples to estimate parameters"
        )
        logger.info(log_message)
        latent = latent[np.random.choice(n_samples, 200000, replace=False)]
        
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type
    )

    log_message = "Fit gaussian mixture model"
    logger.info(log_message)
    gmm.fit(latent)
    labels = gmm.predict(latent)
    
    return gmm.means_, gmm.weights_, latent, labels