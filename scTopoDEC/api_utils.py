import scanpy as sc
import anndata as ad

def run_scTopoDEC_large_data(adata, max_cells=10000, **kwargs):
    """
    Wrapper for scTopoDEC that handles automated subsampling for datasets > 10k cells.
    """
    # Check if the model need to subsample
    if adata.n_obs > max_cells:
        print(f"Dataset has {adata.n_obs} cells. Subsampling to {max_cells} for training...")
        # Use a fixed seed for reproducibility
        adata_train = sc.pp.subsample(adata, n_obs=max_cells, random_state=kwargs.get('random_state', 0), copy=True)
    else:
        adata_train = adata.copy()
    
    # Run the actual scTopoDEC function with the provided keyword arguments
    # Set return_model=True to extract the trained weights
    print("Starting training on subset...")
    network = scTopoDEC(adata_train, return_model=True, copy=False, **kwargs)
    
    # Project the trained network onto the full original dataset
    print("Projecting full dataset using learned weights...")
    network.predict(adata, mode='full', copy=False)
    
    return adata, network