import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from anndata import AnnData


def _log1p(x):
    """
    In-place log1p transformation. 
    Modernized to handle scipy.sparse and numpy arrays consistently.
    """
    if issparse(x):
        # Directly modify the data attribute of the CSR/CSC matrix
        np.log1p(x.data, out=x.data)
    else:
        np.log1p(x, out=x)
    return x


def log1p(data, copy=False):
    if copy:
        data = data.copy()
    if isinstance(data, AnnData):
        _log1p(data.X)
    else:
        _log1p(data)
    return data if copy else None


def get_mean_var(X):
    """
    Calculates mean and variance. 
    Updated to replace .A1 (deprecated) with .flatten() or np.ravel().
    """
    if issparse(X):
        # Ensure it's CSR for efficient math
        if not isinstance(X, csr_matrix):
            X = X.tocsr()
        
        mean = np.array(X.mean(axis=0)).ravel()
        # mean_sq calculated via X^2
        mean_sq = np.array(X.multiply(X).mean(axis=0)).ravel()
    else:
        mean = np.mean(X, axis=0)
        mean_sq = np.mean(np.multiply(X, X), axis=0)

    # Unbiased estimator (N/(N-1))
    n_obs = X.shape[0]
    if n_obs <= 1:
        var = np.zeros_like(mean)
    else:
        var = (mean_sq - mean**2) * (n_obs / (n_obs - 1))
    
    # Clean up negative variance from float precision errors
    var[var < 0] = 0
    return mean, var


def scale_bygroup(adata, groupby=None, max_value=6):
    """
    Standardizes data by group (e.g., by batch).
    Modernized for Pandas 2.0+ and Scanpy 1.10+.
    """
    assert isinstance(adata, AnnData), 'adata must be an AnnData object'
    
    # Efficiently convert only if needed
    if issparse(adata.X):
        adata.X = adata.X.toarray()

    if groupby is not None and groupby in adata.obs.keys():
        # Ensure the groupby column is categorical
        if not isinstance(adata.obs[groupby].dtype, pd.CategoricalDtype):
            adata.obs[groupby] = adata.obs[groupby].astype('category')
            
        categories = adata.obs[groupby].cat.categories
        
        for category in categories:
            # Use boolean indexing for modern Pandas/AnnData
            idx = adata.obs[groupby] == category
            tmp = adata.X[idx].copy()
            
            mean0, var0 = get_mean_var(tmp)
            sd0 = np.sqrt(var0)
            
            # Prevent division by zero
            sd0[sd0 <= 1e-5] = 1e-5
            
            tmp -= mean0
            tmp /= sd0
            
            if max_value is not None:
                tmp = np.clip(tmp, a_min=None, a_max=max_value)
            
            # Assign back to original matrix
            adata.X[idx] = tmp
    else:
        if groupby is not None:
            print(f"Warning: The groupby '{groupby}' does not exist. Scaling across all cells.")
        
        res = adata.X
        mean0, var0 = get_mean_var(res)
        sd0 = np.sqrt(var0)
        sd0[sd0 <= 1e-5] = 1e-5
        
        res -= mean0
        res /= sd0
        
        if max_value is not None:
            res = np.clip(res, a_min=None, a_max=max_value)
        
        adata.X = res
        
    return adata

