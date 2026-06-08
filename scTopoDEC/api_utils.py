import scanpy as sc
from scTopoDEC.io import read_dataset, normalize
from scTopoDEC.api import scTopoDEC

def run_scTopoDEC_large_data(adata, max_cells=2000, leiden_subsampling=False, leiden_resolution=0.5, **kwargs):
    """
    Wrapper for scTopoDEC to handle large single-cell datasets by subsampling a representative 
    training subset before projecting the entire dataset.

    Args:
        adata (AnnData): The full raw single-cell dataset (expected to be an AnnData object).
        max_cells (int): The number of cells to subsample for model training. Defaults to 2000.
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
    # Preprocess full dataset to identify HVGs
    adata_full = adata.copy()
    adata_full = read_dataset(adata_full, check_counts=kwargs.get('check_counts', True), copy=False)

    if kwargs.get('use_hvg', True):
        sc.pp.highly_variable_genes(adata_full, n_top_genes=kwargs.get('n_top_genes', 2000), flavor='seurat_v3')
        adata_full = adata_full[:, adata_full.var.highly_variable]

    # Subsampling
    if adata_full.n_obs > max_cells:
        print(f"Dataset has {adata_full.n_obs} cells. Subsampling to {max_cells}...")
        if leiden_subsampling:
            sc.pp.neighbors(adata_full)
            sc.tl.leiden(adata_full, resolution=leiden_resolution)
            adata_train = sc.pp.subsample(adata_full, groupby='leiden', n_obs=max_cells, 
                                          random_state=kwargs.get('random_state', 0), copy=True)
        else:
            adata_train = sc.pp.subsample(adata_full, n_obs=max_cells, 
                                          random_state=kwargs.get('random_state', 0), copy=True)
    else:
        adata_train = adata_full.copy()

    # Match the genes from subset to the full dataset
    sc.pp.filter_genes(adata_train, min_counts=1)
    adata_full = adata_full[:, adata_train.var_names].copy()

    # Normalize full dataset (required for network.predict)
    adata_full = normalize(adata_full, 
                           filter_min_counts=False,
                           size_factors=kwargs.get('normalize_per_cell', True), 
                           logtrans_input=kwargs.get('log1p', True), 
                           normalize_input=kwargs.get('scale', True))

    # Inject input_size dynamically
    if 'network_kwds' not in kwargs:
        kwargs['network_kwds'] = {}
    kwargs['network_kwds']['input_size'] = adata_train.shape[1]

    # Run training on the subset
    print("Starting training on subset...")
    network = scTopoDEC(
        adata_train, 
        use_hvg=False,            # Data is already filtered to HVGs
        normalize_per_cell=False, # Data is already normalized
        scale=False,              # Data is already scaled
        log1p=False,              # Data is already log-transformed
        check_counts=False,       # Counts have already been checked
        return_model=True, 
        copy=False, 
        **kwargs
    )
    
    # Project using the normalized, HVG-filtered full dataset
    print("Projecting full dataset using learned weights...")
    network.predict(adata_full, mode='clustering', copy=False)
    
    # Map results back to original adata
    adata.obs['stc_cluster'] = adata.obs_names.map(adata_full.obs['stc_cluster'])
    adata.obsm['stc_probs'] = adata_full.obsm['stc_probs']
    adata.obsm['X_stc'] = adata_full.obsm['X_stc']
    
    return adata, network