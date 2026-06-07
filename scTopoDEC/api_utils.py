import scanpy as sc
from scTopoDEC.io import read_dataset, normalize
from scTopoDEC.api import scTopoDEC

def run_scTopoDEC_large_data(adata, max_cells=2000, **kwargs):
    """
     Wrapper for scTopoDEC that handles automated subsampling for datasets > 10k cells.
    """
    # Preprocess full dataset to identify HVGs
    adata_full = adata.copy()
    adata_full = read_dataset(adata_full, check_counts=kwargs.get('check_counts', True), copy=False)
    adata_full.layers["counts"] = adata_full.X.copy()

    if kwargs.get('use_hvg', True):
        sc.pp.highly_variable_genes(adata_full, n_top_genes=kwargs.get('n_top_genes', 2000), flavor='seurat_v3')
        adata_full = adata_full[:, adata_full.var.highly_variable]

    selected_genes = adata_full.var_names

    # Normalize full dataset (required for network.predict)
    adata_full = normalize(adata_full, 
                           filter_min_counts=False,
                           size_factors=kwargs.get('normalize_per_cell', True), 
                           logtrans_input=kwargs.get('log1p', True), 
                           normalize_input=kwargs.get('scale', True))
    
    print(f"Full dataset after HVG selection has {adata_full.n_obs} cells and {adata_full.n_vars} genes.")

    # Subsample
    if adata_full.n_obs > max_cells:
        print(f"Dataset has {adata_full.n_obs} cells. Subsampling to {max_cells}...")
        adata_train = sc.pp.subsample(adata_full, n_obs=max_cells, 
                                      random_state=kwargs.get('random_state', 0), copy=True)
    else:
        adata_train = adata_full.copy()

    # Inject input_size dynamically
    if 'network_kwds' not in kwargs:
        kwargs['network_kwds'] = {}
    kwargs['network_kwds']['input_size'] = len(selected_genes)

    # Run training on the subset
    print("Starting training on subset...")
    network = scTopoDEC(
        adata_train, 
        use_hvg=False,            # Data is already filtered to HVGs
        normalize_per_cell=False, # Data is already normalized
        scale=False,              # Data is already scaled
        log1p=False,              # Data is already log-transformed
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