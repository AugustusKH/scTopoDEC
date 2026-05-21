import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1' 

import csv, tempfile, shutil, random
import anndata
import numpy as np
import pandas as pd
import scanpy as sc

try:
    import keras
except ImportError:
    raise ImportError('scTopoDEC requires Keras v3+. Please follow instructions'
                      ' at https://keras.io/getting_started/ to install.')

try:
    import tensorflow as tf
except ImportError:
    raise ImportError('scTopoDEC requires TensorFlow v2+. Please follow instructions'
                      ' at https://www.tensorflow.org/install/ to install.')

try:
    import gudhi as gd
except ImportError:
    raise ImportError('scTopoDEC requires GUDHI v3+. Please follow instructions'
                      ' at https://gudhi.inria.fr/python/latest/installation.html to install.')

from .io import read_dataset, normalize
from .train import pretrain, train, ramp_train
from .network import network_options, Autoencoder, ZINBAutoencoder, DEC
from .utils import set_reproducibility

tf.keras.backend.clear_session()


def scTopoDEC(adata,                        # single-cell args
        ae_type='dec',
        mode='clustering',
        n_clusters='10',
        n_neighbors=20, 
        resolution=0.8,
        use_hvg=True,         
        n_top_genes=2000,      
        alpha=1.,
        normalize_per_cell=True,
        scale=True,
        log1p=True,
        hidden_size=(256, 64, 32, 64, 256), # network args
        loss_weights=(1, 0.1, 0, 1),
        noise_sd=0.,
        hidden_dropout=0.,
        mask_rate=0.0,
        batchnorm=True,
        activation='relu',
        init='glorot_uniform',
        network_kwds={},
        epochs=300,                         # Train args
        optimizer='adam',
        learning_rate=0.01,
        update_interval=10,
        tol=1e-3,
        ground_truth=None,
        batch_key=None,
        res_ramp=(0.0, 0.1, 0.2, 0.5, 1.0),
        ramp_mode=False,
        cluster_early_stop=True,
        soft_kmean=False,
        training_kwds={},
        pretrain_epochs=200,                # Pretrain args    
        pretrain_optimizer='adam',
        pretrain_learning_rate=0.01,  
        pretraining_kwds={},         
        reduce_lr=20,                       # Both pretrain and train args
        early_stop=30,
        batch_size=128,
        random_state=0,
        threads=None,
        verbose=False,
        return_model=False,
        return_info=False,
        copy=False,
        check_counts=True,
        homology_dim=1,                     # Topology args
        maximum_edge_length=1.5,
        topo_size=64,
        pg_dist='wd',
        order=1.,
        topo_input_mode='eff_res',
        topo_latent_mode='eff_res',
        n_components=30, 
        k=30, 
        t=8,
        train_output_dir=None,              # Model weight save/load args 
        pretrain_output_dir=None,       
        initial_pretrain_weights=None,  
        initial_train_weights=None,     
        save_pretrain_weights=False,     
        save_train_weights=False,        
        ):
    """Single-cell topological deep embedded clustering (scTopoDEC) API.
        
        This package performs single-cell clustering using a ZINB Autoencoder 
        framework. It optimizes a joint loss function comprising:

        1. Reconstruction Loss: ZINB-based denoising of count data.
        2. Clustering Loss: KL-divergence for cluster sharpening (DEC).
        3. Topological Loss: Calculated via persistent homology to preserve 
           global data structure.

        Inputs and parameters:
        ======================
        adata : class:`anndata.AnnData`
            A single-cell object. Must include raw counts in `.X` or `.raw.X` for 
            ZINB modeling.
        ae_type : `str`, optional (default: 'dec')
            Type of the autoencoder architecture:
            - 'ae': standard autoencoder (MSE loss).
            - 'zinb': ZINB autoencoder (probabilistic reconstruction).
            - 'dec': deep embedded clustering (joint ZINB + clustering loss).
        mode : `str`, optional (default: 'clustering')
            Defines the output scope of the analysis:
            - 'denoise': Replaces `adata.X` with denoised/imputed counts.
            - 'latent': Adds low-dimensional representation to `adata.obsm['X_dca']` or
            `adata.obsm['X_dec_latent']`.
            - 'clustering': Adds cluster assignments to `adata.obs['dec_cluster']` and 
            probabilities to `adata.obsm['X_dec_probs']`.
            - 'full': Executes all modes and updates all relevant fields.
        n_clusters : int or str, optional (default: 10)
            The number of clusters to find during the DEC phase. This determines the number 
            of centroids initialized by K-Means and the output dimensions of the clustering 
            layer. If n_cluster = 0, then the model will detect the number of clusters 
            automatically using the Leiden algorithm.
        n_neighbors : int, optional (default: 20) 
            The number of nearest neighbors used to calculate the Leiden algorithm for automatic
            detection of the number of clusters.  
        resolution : float, optional (default: 0.8)
            The resolution parameter for the Leiden algorithm for automatic detection of 
            the number of clusters.
        use_hvg : bool, optional (default: True)
            If True, the model identifies and trains only on Highly Variable Genes (HVGs). 
            This significantly improves training stability, reduces the risk of NaN errors, 
            and speeds up computation by focusing on biological signals rather than technical 
            noise.
        n_top_genes : int, optional (default: 2000)
            The number of top variable genes to keep if use_hvg is True. For datasets with 
            ~30,000 genes, 2,000 is the recommended benchmark for optimal clustering performance.
        loss_weights : `list`, optional (default: (1, 0.1, 0, 1))
            Weights for the joint loss function:
            - index 0: Reconstruction loss (ZINB/MSE).
            - index 1: Clustering loss (KLD).
            - index 2: Soft k-mean clustering loss.
            - index 3: Topological loss (persistent Homology).
        alpha : `float`, optional (default: 1.0)
            Degrees of freedom for Student’s t-distribution in the clustering layer. 
            Must be positive to calculate soft assignments (q).
        normalize_per_cell : `bool`, optional (default: True)
            If True, library size normalization is performed and saved as size factors. 
            The decoder re-introduces these factors to scale the output mean layer.
        scale : `bool`, optional (default: True)
            If True, the encoder input is centered/scaled. Reconstruction loss continues to 
            target the raw counts.
        log1p : `bool`, optional (default: True)
            If True, the input is log-transformed (log(1+x)) for the encoder.
        hidden_size : `tuple` or `list`, optional (default: (256, 64, 32, 64, 256))
            Number of neurons in hidden layers (symmetric for encoder/decoder).
        hidden_dropout : `float`, optional (default: 0.0)
            Dropout rate applied to hidden layers.
        mask_rate : `float`, optional (default: 0.0)
            The fraction of gene expression features to randomly mask (set to 0) 
            during the initial autoencoder pretraining phase. 
            - Set to 0.0 to disable masking and run standard pretraining.
            - Set between 0.1 and 0.4 (e.g., 0.2 for 20% masking) to enable 
              Self-Supervised Masked Gene Modeling (MGM). 
            - This forces the network to learn complex gene co-expression relationships 
              by predicting missing values from unmasked features, heavily mitigating 
              the effects of technical dropout. Masking is automatically disabled 
              during the clustering and evaluation phases.
        batchnorm : `bool`, optional (default: True)
            If True, applies Batch Normalization after each dense layer.
        activation : `str`, optional (default: 'relu')
            Activation function for hidden layers (supports 'PReLU', 'LeakyReLU', etc.).
        init : `str`, optional (default: 'glorot_uniform')
            Weight initialization method.
        network_kwds : `dict`, optional
            Additional arguments passed to the Network class constructor.
        epochs : `int`, optional (default: 300)
            Maximum number of training iterations.
        optimizer : `str`, optional (default: 'adam')
            Optimization algorithm (e.g., 'adam', 'RMSprop').
        learning_rate : `float`, optional (default: 0.001)
            Initial learning rate for the optimizer.
        update_interval : `int`, optional (default: 10)
            The frequency (in epochs) at which the target distribution 'p' is 
            recalculated. This acts as the "self-supervision" signal for DEC; 
            updating too frequently can cause instability, while too rarely 
            can lead to slow convergence.
        tol : `float`, optional (default: 1e-3)
            The convergence threshold for cluster stability. It represents the 
            fraction of cells that change their cluster assignment between 
            consecutive `update_interval` checks. 
            - A value of 0.001 (1e-3) means training stops if fewer than 0.1% 
              of cells change clusters.
            - Lower values (e.g., 1e-5) force the model to reach a higher state 
              of stability, useful for very fine-grained sub-clustering.
        ground_truth : `str` or None, optional (default: None)
            A key in `adata.obs` containing known cell-type labels. If provided, 
            the model will output Accuracy (ACC), Normalized Mutual Info (NMI), 
            and Adjusted Rand Index (ARI) during training to monitor performance.
        batch_key : `str` or None, optional (default: None)
            A key in `adata.obs` containing known multiple sample labels. Assigning the key
            makes the model handles with batch effects.
        res_ramp : `list`, optional (default: (0.0, 0.1, 0.2, 0.5, 1.0))
            A list of scaling factors for the clustering loss weight. This implements 
            a "curriculum learning" strategy where the model first focuses on 
            reconstruction (low values) and gradually increases the pressure to 
            form tight clusters (high values). This prevents the clustering loss 
            from distorting the biological manifold early in training.
        ramp_mode : `bool`, optional (default: False)
            If True, enables the multi-stage iterative ramping process. When 
            enabled, the total `epochs` are divided across the stages defined 
            in `res_ramp`, and the learning rate is annealed at each stage to 
            ensure stable convergence.
        cluster_early_stop : `bool`, optional (default: True)
            If True, enable patience for early stopping in clustering training step. 
        soft_kmean : `bool`, optional (default: False)
            If True, use soft k-mean loss function, else use standard k-mean loss function.
        training_kwds : `dict`, optional
            Additional arguments passed to the training function, i.e. dec_train().
        pretrain_epochs : `int`, optional (default: 200)
            Number of iterations for the initial autoencoder pretraining phase. 
            This ensures the weights are "warm" and the latent space is 
            structured before clustering begins.
        pretrain_optimizer : `str`, optional (default: 'adam')
            The optimization algorithm used specifically for the pretraining 
            phase (e.g., 'adam', 'sgd').
        pretrain_learning_rate : `float`, optional (default: 0.01)
            The learning rate for the pretraining phase. This is typically higher 
            than the clustering learning rate to allow for rapid feature learning.
        reduce_lr : `int`, optional (default: 20)
            Patience for Reducing Learning Rate on plateau.
        early_stop : `int`, optional (default: 30)
            Patience for Early Stopping based on validation loss.
        batch_size : `int`, optional (default: 128)
            Number of samples per gradient update.
        random_state : `int`, optional (default: 0)
            Seed for reproducibility (affects Python, NumPy, and TensorFlow).
        threads : `int` or None, optional (default: None)
            CPU threads for TensorFlow. If None, uses all available cores.
        verbose : `bool`, optional (default: False)
            If True, outputs detailed training progress and model summaries.
        pretraining_kwds : `dict`, optional
            Additional arguments passed to the `.fit()` pretraining process.
        return_model : `bool`, optional (default: False)
            If True, returns the trained network object for further use.
        return_info : `bool`, optional (default: False)
            If True, stores ZINB parameters (dispersion, dropout probabilities) in `adata.obsm`.
        copy : `bool`, optional (default: False)
            If True, returns a modified copy of the AnnData object.
        check_counts : `bool`, optional (default: True)
            Verifies that the input data consists of unnormalized integer counts.
        homology_dim : `int`, optional (default: 1)
            The Betti number/dimension to calculate.
            - 0: Connected components (clusters).
            - 1: Cycles/loops (trajectories/branches).
            - 2: Voids/spheres (globular structures).
        maximum_edge_length : `float`, optional (default: 1.5)
            The filtration cutoff. Limits the distance at which points are connected. 
            Prevents OOM errors by ignoring very long-distance edges.
        topo_size : `integer`, optional (default: 64)
            The number of cells randomly sampled to estimate the density 
            scale, ensuring the loss is computationally efficient and scale-invariant.
        pg_dist : `string`, optional (default: 'wd') 
            Method used to measure persistent diagram distance: 'mse', Mean Squared Error (MSE); 
            'weight_mse', weighted MSE; 'wd', Wasserstein distance.
        order : `float`, optional (default: 1.)
            Wasserstein exponent q (1 <= q < infinity).
        topo_input_mode : `string`, optional (default: 'eff_res')
            The method used to generate the ground-truth topological representation from the 
            input data. Options include coordinate-based point clouds ('pca', 'tsne', 'umap', 'raw') or 
            pre-computed distance matrices ('pca_dist', 'tsne_dist', 'umap_dist', 'knn', 'eff_res', 'diffusion').
        topo_latent_mode : `string`, optional (default: 'eff_res')
            The method used to represent the latent space Z for topological comparison. 
            Use 'raw' for coordinate-based comparison, 'inner_product' for matrix multiplication between Z and 
            its transpose, or 'euclid_dist'/'knn'/'eff_res'/'diffusion' for distance-matrix-based comparison to 
            ensure the operations remain differentiable during training.
        n_components : `int`, optional (default: 30)
            The number of dimensions to retain when using 'pca', 'umap', or their corresponding distance 
            modes for the input representation.
        k : `int`, optional (default: 30)
            The number of nearest neighbors used to construct the adjacency matrix for graph-based modes 
            (e.g., 'knn', 'eff_res', and 'diffusion'). 
        t : `int`, optional (default: 8)
            The diffusion time (number of power iterations) applied to the transition matrix when calculating 
            diffusion distances.
        train_output_dir : `str` or None, optional (default: None)
            Directory path to save final training weights and session metadata.
        pretrain_output_dir : `str` or None, optional (default: None)
            Directory path to save pretraining checkpoints.
        initial_pretrain_weights : `str` or None, optional (default: None)
            Path to a `.weights.h5` file. If provided, the model loads these weights and skips the initial autoencoder 
            pretraining phase.
        initial_train_weights : `str` or None, optional (default: None)
            Path to a `.weights.h5` file. If provided, the model restores these weights to resume or warm-start 
            a clustering session.
        save_pretrain_weights : `bool`, optional (default: False)
            If True, automatically saves the best autoencoder weights to `pretrain_output_dir` during the pretraining phase.
        save_train_weights : `bool`, optional (default: False)
            If True, enables automatic checkpointing of weights during clustering training, saving the 'best' state to 
            `train_output_dir`.

        Outputs
        =======
        Depends on `mode`, `ae_type`, and `copy` parameters.
        
        If `copy=True`:
            Returns an AnnData object with the following fields:
            `adata.X` : `numpy.ndarray`
                Imputed/denoised expression values (if mode is 'denoise' or 'full').
            `adata.obsm['X_dca']` or `adata.obsm['X_dec_latent']` : `numpy.ndarray`
                Low-dimensional latent representation (if mode is 'latent' or 'full').
            `adata.obs['dec_cluster']` : `pandas.Series`
                Predicted cluster assignments (if mode is 'clustering' or 'full' and 
                ae_type is 'dec').
            `adata.obsm['X_dec_probs']` : `numpy.ndarray`
                Soft cluster assignment probabilities (q) (if mode is 'clustering' or 
                'full' and ae_type is 'dec').
            `adata.obsm['X_dec_dispersion']` & `adata.obsm['X_dec_dropout']` : `numpy.ndarray`
                ZINB model parameters (if return_info=True).
        
        If `copy=False`:
            Updates the input `adata` object in-place and returns `None`.

        If `return_model=True`:
            Returns the trained `scTopoDEC` network object for downstream weight analysis 
            or further fine-tuning.
    """

    # 1. Parameter Validation
    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'
    assert ae_type in network_options.keys(), '%s is not a valid network architecture.' % ae_type
    
    # Mode validation based on architecture type
    if ae_type in ('ae', 'zinb'):
        assert mode in ('denoise', 'latent', 'full'), '%s is not valid for imputation.' % mode
    else:
        assert mode in ('denoise', 'latent', 'clustering', 'full'), '%s is not valid for clustering.' % mode

    # 2. Setup reproducibility and threads
    if threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)

    set_reproducibility(seed=random_state)

    if verbose:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            details = tf.config.experimental.get_device_details(gpus[0])
            print(f"Reproducibility Lock: Enabled. GPU Detected: {details.get('device_name', 'Unknown')}")

            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        else:
            print("Reproducibility Lock: Enabled. Running on CPU.")

    # 3. Data Preprocessing
    adata = adata.copy() if copy else adata
    adata_train = adata.copy()
    adata_train = read_dataset(adata_train, check_counts=check_counts, copy=False)

    should_use_hvg = (use_hvg and ae_type == 'dec')
    
    if should_use_hvg:
        print(f"DEC mode: Selecting top {n_top_genes} highly variable genes...")
        sc.pp.highly_variable_genes(adata_train, n_top_genes=n_top_genes, flavor='seurat_v3')
        adata_train = adata_train[:, adata_train.var.highly_variable]
    else:
        print(f"{ae_type.upper()} mode: Using full dataset ({adata.n_vars} genes).")

    adata_train = normalize(adata_train, 
                      size_factors=normalize_per_cell, 
                      logtrans_input=log1p, 
                      normalize_input=scale)
    
    # 4. Handle batch one-hot encoding
    n_batch = 0
    if batch_key is not None:
        if batch_key not in adata_train.obs.columns:
            raise ValueError(f"batch_key '{batch_key}' not found in adata.obs")
        
        # Convert batch categories to one-hot matrix
        batch_matrix = pd.get_dummies(adata_train.obs[batch_key]).values
        adata_train.obsm['batch_onehot'] = batch_matrix.astype(np.float32)
        n_batch = batch_matrix.shape[1]
        print(f"Batch correction enabled: {n_batch} batches detected.")

    # 5. Model Initialization
    model_kwargs = {
        'input_size': adata_train.n_vars,
        'output_size': adata_train.n_vars,
        'n_batch': n_batch,
        'hidden_size': hidden_size,
        'hidden_dropout': hidden_dropout,
        'noise_sd': noise_sd,
        'mask_rate': mask_rate,
        'batchnorm': batchnorm,
        'activation': activation,
        'init': init,
        'debug': verbose,
        **network_kwds
    }

    if ae_type == 'dec':
        n_cl = int(n_clusters)
        if n_cl > 0:
            model_kwargs['n_clusters'] = n_cl
            training_kwds['auto_detect'] = False
        else:
            model_kwargs['n_clusters'] = 1 
            training_kwds['auto_detect'] = True
            training_kwds['n_neighbors'] = n_neighbors
            training_kwds['resolution'] = resolution

        model_kwargs['alpha'] = alpha

    network = network_options[ae_type](**model_kwargs)
    network.build()

    # 5. Execution Logic
    if ae_type == 'dec':
        # Clustering pathway
        if ramp_mode:
            ramp_train(adata_train, network, 
                       random_state=random_state,
                       optimizer=optimizer, 
                       learning_rate=learning_rate, 
                       epochs=epochs, 
                       batch_size=batch_size,
                       loss_weights=loss_weights,
                       update_interval=update_interval,
                       tol=tol,
                       early_stop_patience=early_stop,
                       cluster_early_stop=cluster_early_stop,
                       soft_kmean=soft_kmean,
                       ground_truth=ground_truth,
                       res_ramp=res_ramp,
                       pretrain_epochs=pretrain_epochs,
                       pretrain_optimizer=pretrain_optimizer,
                       pretrain_learning_rate=pretrain_learning_rate,
                       homology_dim=homology_dim, 
                       maximum_edge_length=maximum_edge_length,
                       topo_size=topo_size,
                       pg_dist=pg_dist, order=order,
                       topo_input_mode=topo_input_mode, 
                       topo_latent_mode=topo_latent_mode, 
                       n_components=n_components, k=k, t=t,
                       train_output_dir=train_output_dir,
                       initial_train_weights=initial_train_weights,
                       save_train_weights=save_train_weights,
                       pretrain_output_dir=pretrain_output_dir,
                       initial_pretrain_weights=initial_pretrain_weights,
                       save_pretrain_weights=save_pretrain_weights,
                       n_batch=n_batch,
                       verbose=verbose,
                       **training_kwds)
        else:
            train(adata_train, network, 
                  random_state=random_state,
                  optimizer=optimizer, 
                  learning_rate=learning_rate, 
                  epochs=epochs, 
                  batch_size=batch_size,
                  loss_weights=loss_weights,
                  update_interval=update_interval,
                  tol=tol,
                  reduce_lr_patience=reduce_lr,
                  early_stop_patience=early_stop,
                  cluster_early_stop=cluster_early_stop,
                  soft_kmean=soft_kmean,
                  homology_dim=homology_dim, 
                  maximum_edge_length=maximum_edge_length,
                  ground_truth=ground_truth,
                  pretrain_epochs=pretrain_epochs,
                  pretrain_optimizer=pretrain_optimizer,
                  pretrain_learning_rate=pretrain_learning_rate,
                  topo_size=topo_size,
                  pg_dist=pg_dist, order=order,
                  topo_input_mode=topo_input_mode, 
                  topo_latent_mode=topo_latent_mode, 
                  n_components=n_components, k=k, t=t,
                  train_output_dir=train_output_dir,
                  initial_train_weights=initial_train_weights,
                  save_train_weights=save_train_weights,
                  pretrain_output_dir=pretrain_output_dir,
                  initial_pretrain_weights=initial_pretrain_weights,
                  save_pretrain_weights=save_pretrain_weights,
                  n_batch=n_batch,
                  verbose=verbose,
                  **training_kwds)
    else:
        # Imputation pathway
        if initial_pretrain_weights and os.path.exists(initial_pretrain_weights):
            print(f"Loading weights from {initial_pretrain_weights}...")
            network.load_weights(initial_pretrain_weights)
        else:
            pretrain(adata_train, network, 
                    output_dir=pretrain_output_dir, 
                    save_weights=save_pretrain_weights,
                    optimizer=pretrain_optimizer, 
                    learning_rate=pretrain_learning_rate, 
                    epochs=pretrain_epochs, 
                    batch_size=batch_size,
                    reduce_lr=reduce_lr,
                    early_stop=early_stop,
                    verbose=verbose,
                    **pretraining_kwds)
            
    # 6. Save of the entire object (Metadata + Weights)
    if train_output_dir is not None:
        print(f"Saving final model state to {train_output_dir}...")
        network.file_path = train_output_dir
        network.save()

    # 7. Inference and results
    network.predict(adata_train, mode=mode, return_info=return_info, copy=False)

    # 8. Return outputs
    print('\nAlgorithm runs successfully!')
    if ae_type == 'dec':
        adata.obs['stc_cluster'] = adata.obs_names.map(adata_train.obs['stc_cluster'])
        adata.obsm['stc_probs'] = adata_train.obsm['stc_probs'] 
        adata.obsm['stc'] = adata_train.obsm['stc']
        
        if return_model:
            return (adata, network) if copy else network
        else:
            return adata if copy else None

    else:
        if return_model:
            return (adata_train, network) if copy else network
        else:
            return adata_train if copy else None
        
    

