import pickle, os, numbers, h5py
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import tensorflow as tf

from keras.utils import Sequence
from sklearn.model_selection import train_test_split


class AnnSequence(Sequence):
    def __init__(self, matrix, batch_size, sf=None):
        self.matrix = matrix
        self.size_factors = sf if sf is not None else np.ones((self.matrix.shape[0], 1), dtype=np.float32)
        self.batch_size = batch_size
        self.indices = np.arange(len(self.matrix))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.matrix) / float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        curr_idx = self.indices[start:end]
        batch = self.matrix[curr_idx]
        batch_sf = self.size_factors[curr_idx]

        # Only convert the small micro-batch to dense to save memory
        if sp.sparse.issparse(batch):
            batch = batch.toarray()
        elif hasattr(batch, "toarray"):
            batch = batch.toarray()
        
        batch = np.asarray(batch, dtype=np.float32)

        # return an (X, Y) pair
        return {'count': batch, 'size_factors': batch_sf}, batch
    

class SparseDataGenerator(Sequence):
    """Keras Sequence generator for large sparse data."""
    def __init__(self, adata, indices, batch_size, use_raw_as_output=True, output_subset=None):
        self.adata = adata
        self.indices = indices
        self.batch_size = batch_size
        self.use_raw_as_output = use_raw_as_output
        self.output_subset = output_subset
        self.has_batch = 'batch_onehot' in adata.obsm

        if self.output_subset:
            self.gene_idx = [self.adata.var_names.get_loc(x) for x in self.output_subset]
        else:
            self.gene_idx = None
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Input Count Matrix
        raw_batch_x = self.adata.X[batch_indices]
        if sp.sparse.issparse(raw_batch_x):
            raw_batch_x = raw_batch_x.toarray()
        x_c = raw_batch_x.astype(np.float32)

        # Size Factors
        x_s = self.adata.obs.size_factors.values[batch_indices].astype(np.float32)

        inputs = {'count': x_c, 'size_factors': x_s}

        # Batch Covariates
        if self.has_batch:
            inputs['batch'] = self.adata.obsm['batch_onehot'][batch_indices].astype(np.float32)

        # Target Matrix
        if self.use_raw_as_output:
            raw_batch_y = self.adata.raw.X[batch_indices]
        else:
            raw_batch_y = self.adata.X[batch_indices]

        if self.output_subset:
            raw_batch_y = raw_batch_y[:, self.gene_idx]

        if sp.sparse.issparse(raw_batch_y):
            raw_batch_y = raw_batch_y.toarray()
            
        target = raw_batch_y.astype(np.float32)

        return inputs, target

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def is_raw_counts(matrix):
    """Checks if a matrix contains unnormalized integer counts."""
    # Check a small subset for speed
    subset = matrix[:10, :100] 
    if sp.sparse.issparse(subset):
        data = subset.data
    else:
        data = subset
    
    # Check if values are non-negative integers
    return np.all(data >= 0) and np.all(np.equal(np.mod(data, 1), 0))


def spare_to_dense_count(adata):
    # Densify the active matrix (.X)
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray() 

    # Densify the raw matrix
    if adata.raw is not None:
        raw_dense = adata.raw.to_adata()
        if hasattr(raw_dense.X, "toarray"):
            raw_dense.X = raw_dense.X.toarray()
        adata.raw = raw_dense

    # Densify all layers
    for layer in list(adata.layers.keys()):
        if hasattr(adata.layers[layer], "toarray"):
            adata.layers[layer] = adata.layers[layer].toarray()

    return adata


def read_dataset(adata, transpose=False, test_split=False, copy=False, check_counts=True, spare_to_dense=False):
    if isinstance(adata, sc.AnnData):
        adata = adata.copy() if copy else adata
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    if check_counts:
        # Step 1: Check if current X is raw
        if is_raw_counts(adata.X):
            print("Confirmed: adata.X contains raw counts.")
            if spare_to_dense:
                adata = spare_to_dense_count(adata)
        else:
            print("Notice: adata.X appears normalized. Checking 'counts' layer...")
            
            # Step 2: Check if 'counts' layer exists and is raw
            if "counts" in adata.layers and is_raw_counts(adata.layers["counts"]):
                print("Success: Found raw data in 'counts' layer. Swapping to adata.X.")
                # Optional: Backup the normalized X to a different layer
                adata.layers["normalized"] = adata.X.copy()
                # Move raw counts to X for the preprocessing functions
                adata.X = adata.layers["counts"].copy()
                if spare_to_dense:
                    adata = spare_to_dense_count(adata)
            # Step 3: Check if .raw exists and is raw
            elif adata.raw is not None and is_raw_counts(adata.raw.X):
                print("Success: Found raw data in adata.raw.X. Restoring AnnData from raw state.")
                adata = adata.raw.to_adata()
                if spare_to_dense:
                    adata = spare_to_dense_count(adata)
            else:
                # Step 4: Critical Failure
                raise ValueError(
                    "Error: No raw counts found. ZINB loss requires integer counts. "
                    "Ensure raw data is in adata.X or adata.layers['counts']."
                )

    if transpose: 
        adata = adata.transpose()
        # Safe for sometimes libraries return a sparse view
        if hasattr(adata.X, "toarray") and spare_to_dense:
            adata.X = adata.X.toarray()

    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    adata.raw = adata.copy()

    if size_factors:
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        sc.pp.normalize_total(adata, target_sum=None)
        adata.obs['size_factors'] = adata.obs['total_counts'] / np.median(adata.obs['total_counts'])
    else:
        adata.obs['size_factors'] = 1.0

    # Clip values to avoid log(0)
    if sp.sparse.issparse(adata.X):
        # Clipping sparse structures without breaking compression
        adata.X.data = np.clip(adata.X.data, 1e-10, 1e6)
    else:
        adata.X = np.clip(adata.X, 1e-10, 1e6)

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def data_compression(adata, pca=True, knn=True, tsne=True, umap=True, n_components=30, k=15):
    if pca:
        sc.tl.pca(adata)

    if knn:
        sc.pp.neighbors(adata, n_neighbors=k, n_pcs=n_components)

    if tsne:
        sc.tl.tsne(adata, n_components=n_components)

    if umap:
        sc.tl.umap(adata, n_components=n_components)

    return adata


def read_genelist(filename):
    with open(filename, 'rt') as f:
        genelist = [line.strip() for line in f if line.strip()]
    genelist = list(set(genelist))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    print('dca: Subset of {} genes will be denoised.'.format(len(genelist)))

    return genelist


def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')


def read_pickle(inputfile):
    with open(inputfile, "rb") as f:
        return pickle.load(f)


def h5_to_adata(path):
    with h5py.File(path, 'r') as f:
        # Create the AnnData object with the noisy counts
        adata = sc.AnnData(X=f['X'][:].astype(np.float32))
        
        # Add Ground Truth labels to .obs
        adata.obs['group'] = f['Y'][:].astype(str)
        
        # Add True Counts to .layers
        adata.layers['true_counts'] = f['true_counts'][:].astype(np.float32)
        
        # Set gene and cell names
        adata.var_names = f['var_names'][:].astype(str)
        adata.obs_names = f['obs_names'][:].astype(str)
        
    return adata