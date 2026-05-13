from ..datasets import pbmc_processed
from ..models.desc import train
import scanpy as sc

def run_desc_test():
    """
    Standard test for DESC integration.
    Updated for modern Python 3.12+ and TensorFlow 2.16+ standards.
    """
    print('--- Starting DESC Package Test ---')
    
    # 1. Load sample dataset
    # Ensure pbmc_processed() returns a clean AnnData object
    adata = pbmc_processed()
    
    # 2. Pre-check: Ensure data is scaled or preprocessed if DESC requires it
    # Modern DESC often expects log-normalized and scaled data
    if "n_counts" not in adata.obs:
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    # 3. Execute DESC training
    # Note: resolution is used to initialize the cluster centers
    # dims: [input_dim, hidden1, hidden2, latent_dim] 
    # If adata.n_vars is 2000, dims[0] should be 2000 or the function must handle it.
    try:
        adata = train(
            adata, 
            dims=[adata.n_vars, 64, 16], 
            louvain_resolution=[0.1], # Modern DESC often takes a list for multi-resolution
            n_neighbors=10,
            batch_size=256,
            use_gpu=True
        )
        
        # 4. Validation
        # Check if 'desc_0.1' (or similar) is in adata.obs
        cluster_key = 'desc_0.1'
        if cluster_key in adata.obs:
            n_clusters = len(adata.obs[cluster_key].unique())
            print(f'Test Passed! Found {n_clusters} clusters.')
        else:
            print('Training finished but clustering labels not found in adata.obs.')
            
    except Exception as e:
        print(f'Test Failed with error: {e}')
        raise e

    return None