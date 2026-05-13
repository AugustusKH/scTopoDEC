"""
Keras implement Deep learning enables accurate clustering and batch effect removal in single-cell RNA-seq analysis
"""
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Layer, InputSpec
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import scanpy as sc
from time import time as get_time

# Import the updated SAE class
try:
    from .SAE import SAE
except ImportError:
    from SAE import SAE


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        # Modern weight initialization
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim), 
            initializer='glorot_uniform', 
            name='clusters'
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution with degree alpha, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def get_config(self):
        config = {'n_clusters': self.n_clusters, 'alpha': self.alpha}
        base_config = super().get_config()
        return {**base_config, **config}


class ClusteringLayerGaussian(ClusteringLayer):
    """
    Alternative clustering layer using a Gaussian kernel.
    Generally less robust for scRNA-seq than the t-distribution.
    """
    def __init__(self, n_clusters, weights=None, alpha=1.0, sigma=1.0, **kwargs):
        super().__init__(n_clusters, weights, alpha, **kwargs)
        self.sigma = sigma
    
    def call(self, inputs, **kwargs):
        # Calculate squared Euclidean distance
        # Shape: (batch_size, n_clusters)
        dist_sq = K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2)
        
        # Gaussian kernel: exp(-dist^2 / 2*sigma^2)
        q = K.exp(-dist_sq / (2.0 * self.sigma**2))
        
        # Normalize to get soft labels (sum to 1 per sample)
        q = q / K.sum(q, axis=1, keepdims=True)
        return q


class DescModel(object):
    def __init__(self, dims, x, alpha=1.0, tol=0.005, init='glorot_uniform',
                 louvain_resolution=0.8, n_neighbors=15, pretrain_epochs=300,
                 epochs_fit=4, batch_size=256, random_seed=201809,
                 activation='relu', actincenter="tanh", drop_rate_SAE=0.2,
                 is_stacked=True, use_earlyStop=True, use_ae_weights=False,
                 save_dir="result_tmp"):

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.dims = dims
        self.x = x.astype('float32')
        self.alpha = alpha
        self.tol = tol
        self.resolution = louvain_resolution
        self.n_neighbors = n_neighbors
        self.pretrain_epochs = pretrain_epochs
        self.epochs_fit = epochs_fit
        self.batch_size = batch_size
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        # Initialize/Load Autoencoder
        self.pretrain(init, activation, actincenter, drop_rate_SAE, is_stacked, use_earlyStop, use_ae_weights)
        

    def pretrain(self, init, activation, actincenter, drop_rate, is_stacked, use_earlyStop, use_ae_weights):
        try:
            from .SAE import SAE # Try relative import first
        except (ImportError, ValueError):
            from SAE import SAE  # Fallback for direct script execution

        sae = SAE(dims=self.dims, act=activation, drop_rate=drop_rate, batch_size=self.batch_size,
                  actincenter=actincenter, init=init, use_earlyStop=use_earlyStop, save_dir=self.save_dir)
        
        ae_weights_path = os.path.join(self.save_dir, 'ae_weights.weights.h5')
        
        if use_ae_weights and os.path.isfile(ae_weights_path):
            print(f"Loading AE weights: {ae_weights_path}")
            sae.autoencoders.load_weights(ae_weights_path)
        else:
            sae.fit(self.x, epochs=self.pretrain_epochs)
            sae.autoencoders.save_weights(ae_weights_path)

        self.encoder = sae.encoder

        # --- LOUVAIN INITIALIZATION ---
        print(f"Initializing centroids using Louvain (resolution={self.resolution})...")
        features = self.encoder.predict(self.x)
        adata0 = sc.AnnData(features)
        
        # Subsample if extremely large to save Louvain memory
        if adata0.n_obs > 200000:
            idx = np.random.choice(adata0.n_obs, 200000, replace=False)
            adata0 = adata0[idx].copy()

        sc.pp.neighbors(adata0, n_neighbors=self.n_neighbors, use_rep="X")
        
        # Explicitly calling louvain (requires 'louvain' package)
        sc.tl.louvain(adata0, resolution=self.resolution)
        
        y_pred_init = adata0.obs['louvain'].astype(int).values
        self.init_pred = np.copy(y_pred_init)
        
        # Calculate cluster centroids
        centroids = []
        for i in sorted(np.unique(y_pred_init)):
            centroids.append(np.mean(adata0.X[y_pred_init == i], axis=0))
        
        self.n_clusters = len(centroids)
        self.init_centroid = [np.array(centroids)]
        print(f"Detected {self.n_clusters} clusters via Louvain.")

        # Build Clustering Model
        clustering_layer = ClusteringLayer(self.n_clusters, weights=self.init_centroid, name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)
        

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T


    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)
        

    def fit(self, maxiter=2000):
        """Deep Clustering Training Loop."""
        y_pred_last = np.copy(self.init_pred)
        
        for ite in range(int(maxiter)):
            # Update target distribution p every 'update_interval'
            q = self.model.predict(self.x, verbose=0)
            p = self.target_distribution(q)
            
            y_pred = q.argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)

            if ite > 0 and delta_label < self.tol:
                print(f'Reached tolerance threshold ({delta_label} < {self.tol}). Stopping.')
                break

            print(f'Iteration {ite}: delta_label = {delta_label:.4f}')
            
            # Train on full data for a few epochs (Fine-tuning phase)
            self.model.fit(self.x, p, epochs=self.epochs_fit, batch_size=self.batch_size, verbose=0)

        return self.encoder.predict(self.x), self.model.predict(self.x)
         

if __name__ == "__main__":
    import argparse
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from time import time as get_time
    from tensorflow.keras.optimizers import SGD

    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description='DescModel class test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=30, type=int) # Small for testing
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--tol', default=0.005, type=float)
    parser.add_argument('--save_dir', default='result_tmp')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 2. Modernized MNIST Loader
    def load_mnist(sample_size=10000):
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        
        # Flatten and Normalize
        x = x.reshape((x.shape[0], -1)).astype('float32') / 255.0
        
        print(f'MNIST samples loaded: {x.shape}')
        id0 = np.random.choice(x.shape[0], sample_size, replace=False)
        return x[id0], y[id0]

    # Force CPU for integration test
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # 3. Data Preparation
    x, y = load_mnist(sample_size=10000)
    dims = [x.shape[-1], 64, 32]

    # 4. Initialize DESC
    # Note: louvain_resolution determines initial cluster count
    desc = DescModel(
        dims=dims,
        x=x,
        louvain_resolution=0.8,
        use_ae_weights=False, # Set to False first to ensure pretraining works
        pretrain_epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

    desc.model.summary()

    # 5. Training
    t0 = get_time()
    # KLD (Kullback-Leibler Divergence) is the core loss for clustering
    desc.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='kld')
    
    # Embedded_z: The latent manifold
    # q_pred: The soft-assignment probabilities
    Embedded_z, q_pred = desc.fit(maxiter=args.maxiter)
    
    # 6. Result Processing
    # Convert probabilities (q) into hard labels
    y_pred_labels = q_pred.argmax(axis=1)
    
    obs_info = pd.DataFrame()
    obs_info["y_true"] = y.astype(str)
    obs_info["y_pred"] = y_pred_labels.astype(str)
    
    # Wrap in AnnData for downstream Scanpy visualization
    adata = sc.AnnData(x, obs=obs_info)
    adata.obsm["X_Embeded_z"] = Embedded_z
    
    print(f'--- Clustering time: {get_time() - t0:.2f} seconds ---')
    
    # Optional: Quick check of accuracy if ground truth exists
    from sklearn.metrics import normalized_mutual_info_score as nmi
    print(f'NMI Score: {nmi(y, y_pred_labels):.4f}')
    
    
