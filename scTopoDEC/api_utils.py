import os
import numpy as np
import pandas as pd
import scanpy as sc
from scTopoDEC.io import read_dataset, normalize
from scTopoDEC.api import scTopoDEC
from scTopoDEC.network import network_options
from scTopoDEC.train import pretrain
from scTopoDEC.utils import set_reproducibility, estimate_optimal_noise

def run_scTopoDEC_large_data(adata, max_cells=2000, leiden_subsampling=False, leiden_resolution=0.5, **kwargs):
    """
    Wrapper for scTopoDEC to handle large single-cell datasets. 
    It pretrains the autoencoder on the full dataset for global manifold learning, 
    then performs topological DEC clustering on a representative subsample, 
    and finally projects the entire dataset.

    Args:
        adata (AnnData): The full raw single-cell dataset (expected to be an AnnData object).
        max_cells (int): The number of cells to subsample for DEC model training. Defaults to 2000.
        leiden_subsampling (bool): If True, performs stratified subsampling using Leiden clusters 
                                   to ensure diverse cell representation. Defaults to False.
        leiden_resolution (float): Resolution parameter for the Leiden clustering if 
                                   leiden_subsampling is enabled. Defaults to 0.5.
        **kwargs: Arbitrary keyword arguments passed directly to the core scTopoDEC API 
                  (e.g., n_clusters, loss_weights, hidden_size).

    Returns:
        adata (AnnData): The input AnnData object, now containing the following fields:
                         - adata.obs['stc_cluster']: Final cluster assignments.
                         - adata.obsm['stc_probs']: Cluster membership probabilities.
                         - adata.obsm['X_stc']: The learned topological latent embedding.
        network (object): The trained scTopoDEC network object containing model weights 
                          and projection methods.
    """
    # ========================================================
    # DETERMINISM ENFORCEMENT
    # ========================================================
    seed_val = kwargs.get('random_state', 0)
    set_reproducibility(seed=seed_val)

    # 1. Preprocess full dataset
    adata_full = adata.copy()
    adata_full = read_dataset(adata_full, check_counts=kwargs.get('check_counts', True), copy=False)

    # 2. Identify HVGs on the full dataset
    if kwargs.get('use_hvg', True):
        sc.pp.highly_variable_genes(adata_full, n_top_genes=kwargs.get('n_top_genes', 2000), flavor='seurat_v3')
        adata_full = adata_full[:, adata_full.var.highly_variable]

    # 3. Normalize the full dataset 
    adata_full = normalize(adata_full, 
                           filter_min_counts=False,
                           size_factors=kwargs.get('normalize_per_cell', True), 
                           logtrans_input=kwargs.get('log1p', True), 
                           normalize_input=kwargs.get('scale', True))

    # ========================================================
    # PHASE 0: BATCH HANDLING AND SUBSAMPLING
    # ========================================================
    # Handle batch one-hot encoding on the full dataset 
    batch_key = kwargs.get('batch_key', None)
    n_batch = 0
    if batch_key is not None:
        if batch_key not in adata_full.obs.columns:
            raise ValueError(f"batch_key '{batch_key}' not found in adata.obs")
        
        # Convert batch categories to one-hot matrix
        batch_matrix = pd.get_dummies(adata_full.obs[batch_key]).values
        adata_full.obsm['batch_onehot'] = batch_matrix.astype(np.float32)
        n_batch = batch_matrix.shape[1]
        print(f"Batch correction enabled: {n_batch} batches detected.")

    # Subsampling (adata_train will now safely inherit the batch_onehot matrix)
    if adata_full.n_obs > max_cells:
        print(f"Dataset has {adata_full.n_obs} cells. Subsampling to {max_cells}...")
        if leiden_subsampling:
            sc.pp.neighbors(adata_full)
            sc.tl.leiden(adata_full, resolution=leiden_resolution)
            adata_train = sc.pp.subsample(adata_full, groupby='leiden', n_obs=max_cells, 
                                          random_state=seed_val, copy=True)
        else:
            adata_train = sc.pp.subsample(adata_full, n_obs=max_cells, 
                                          random_state=seed_val, copy=True)
    else:
        adata_train = adata_full.copy()

    # Inject input_size dynamically based on HVGs
    if 'network_kwds' not in kwargs:
        kwargs['network_kwds'] = {}
    kwargs['network_kwds']['input_size'] = adata_full.shape[1]

    # Dynamic noise estimation if not provided by user
    noise_sd = kwargs.get('noise_sd', None)
    n_clusters = kwargs.get('n_clusters', 0)
    verbose_flag = kwargs.get('verbose', True)
    
    if noise_sd is None:
        print("\n--- Phase 0: Dynamic Noise Estimation ---")
        print("noise_sd is None: Dynamically estimating optimal noise based on dataset separability...")
        guess_k = int(n_clusters) if int(n_clusters) > 0 else None
        
        noise_sd = estimate_optimal_noise(
            adata_train, 
            n_clusters_guess=guess_k,
            resolution=kwargs.get('resolution', 0.8) 
        )
        print(f"Estimated optimal noise_sd: {noise_sd:.4f}")
        kwargs['noise_sd'] = noise_sd

    # ========================================================
    # PHASE 1: GLOBAL PRETRAINING ON FULL DATASET
    # ========================================================
    print(f"\n--- Phase 1: Global Pretraining ({adata_full.n_obs} cells) ---")

    # Safely consolidate network arguments without overwriting
    net_kwargs = {
        'input_size': adata_full.shape[1],
        'output_size': adata_full.shape[1],
        'batch_size': kwargs.get('batch_size', 256),
        'hidden_size': kwargs.get('hidden_size', (256, 32, 256)),
        'hidden_dropout': kwargs.get('hidden_dropout', 0.05),
        'mask_rate': kwargs.get('mask_rate', 0.0),
        'batchnorm': kwargs.get('batchnorm', True),
        'activation': kwargs.get('activation', 'relu'),
        'init': kwargs.get('init', 'glorot_uniform'),
        'debug': verbose_flag
    }
    # Append any user-provided network_kwds
    net_kwargs.update(kwargs.get('network_kwds', {}))
    
    # Initialize the architecture
    n_cl = int(n_clusters) if int(n_clusters) > 0 else 0
    network = network_options['dec'](
        n_clusters=n_cl,
        noise_sd=kwargs['noise_sd'], 
        **net_kwargs
    )
    # Build the network and pass the batch dimension explicitly
    network.build(n_batch=n_batch)
    
    # Temporarily point the network to the Autoencoder only for pretraining
    full_dec_model = network.model
    network.model = network.zinb_ae 

    # Pretrain Autoencoder
    pretrain(adata_full, 
             network, 
             epochs=kwargs.get('pretrain_epochs', 800),
             learning_rate=kwargs.get('pretrain_learning_rate', 1e-3),
             reduce_lr=kwargs.get('reduce_lr', 20),
             early_stop=kwargs.get('early_stop', 30),
             use_raw_as_output=True,
             batch_size=kwargs.get('batch_size', 256),
             verbose=verbose_flag
    )
             
    # Restore the full DEC model for subsequent clustering
    network.model = full_dec_model
             
    # Save global weights to bridge to the main clustering API
    os.makedirs("stc_weights", exist_ok=True)
    global_weights_path = "stc_weights/global_pretrain_weights.weights.h5"
    network.save_weights(global_weights_path)
    
    # Instruct scTopoDEC to load these weights instead of pretraining again
    kwargs['initial_pretrain_weights'] = global_weights_path

    # ========================================================
    # PHASE 2: DEC CLUSTERING ON SUBSAMPLED DATASET
    # ========================================================
    print(f"\n--- Phase 2: Topological Clustering ({adata_train.n_obs} cells) ---")
    # Since normalization was already applied to the data in Phase 1, we force the API to skip this step entirely.
    import scTopoDEC.api as stc_api
    
    original_filter_genes = sc.pp.filter_genes
    original_api_normalize = getattr(stc_api, 'normalize', None)
    
    sc.pp.filter_genes = lambda *args, **kw: None
    if original_api_normalize is not None:
        stc_api.normalize = lambda adata, **kw: adata

    try:
        network = scTopoDEC(
            adata_train, 
            use_hvg=False,            
            normalize_per_cell=False, 
            scale=False,              
            log1p=False,              
            check_counts=False,       
            return_model=True, 
            copy=False, 
            **kwargs
        )
    finally:
        sc.pp.filter_genes = original_filter_genes
        if original_api_normalize is not None:
            stc_api.normalize = original_api_normalize
    
    # ========================================================
    # PHASE 3: PROJECT FULL DATASET
    # ========================================================
    print(f"\n--- Phase 3: Projecting Full Dataset ({adata_full.n_obs} cells) ---")
    network.predict(adata_full, mode='clustering', copy=False)
    
    # Map results back to original adata
    adata.obs['stc_cluster'] = adata.obs_names.map(adata_full.obs['stc_cluster'])
    adata.obsm['stc_probs'] = adata_full.obsm['stc_probs']
    adata.obsm['X_stc'] = adata_full.obsm['X_stc']
    
    return adata, network