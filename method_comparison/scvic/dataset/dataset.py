import logging
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp_sparse
import torch
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Custom Mock Adapters (Direct Scanpy integration - Standalone)
# ==============================================================================

class MockGeneDataset:
    """
    Extracts flat array features directly from a scanpy AnnData object
    to mock the legacy internal metadata schema required by scVIC.
    """
    def __init__(self, adata: sc.AnnData, labels_obs_key: str, batch_obs_key: str):
        # Safely capture sparse or dense count matrices
        if sp_sparse.issparse(adata.X):
            self.X = adata.X.toarray().astype(np.float32)
        else:
            self.X = adata.X.astype(np.float32)
            
        # Standardize categorical cells and groups to flat tracking integers
        le_labels = LabelEncoder()
        self.labels = le_labels.fit_transform(adata.obs[labels_obs_key]).reshape(-1, 1)
        self.n_labels = len(le_labels.classes_)
        self.cell_types = np.array(le_labels.classes_, dtype=str)
        
        le_batch = LabelEncoder()
        self.batch_indices = le_batch.fit_transform(adata.obs[batch_obs_key]).reshape(-1, 1)
        self.n_batches = len(le_batch.classes_)
        
        # Core dimensionality vectors
        self.nb_genes = self.X.shape[1]
        self.gene_names = np.array(adata.var_names, dtype=str)
        
        # Extract mean and variance statistics matching standard scvi-tools baseline logs
        self.local_means = np.mean(self.X, axis=1, keepdims=True)
        self.local_vars = np.var(self.X, axis=1, keepdims=True)
        self.norm_X = self.X / (self.local_means + 1e-8)
        self.corrupted_X = self.X.copy()
        
        # Explicitly initialize placeholders to satisfy load_dataset_from_scVI downstream transfers
        self.protected_attributes = []
        self.dataset_versions = {}
        self.gene_attribute_names = []
        self.cell_attribute_names = []
        self.cell_categorical_attribute_names = []
        self.attribute_mappings = {}
        self.cell_measurements_col_mappings = {}


class MockExpressionDataset:
    """
    Acts as a lightweight container proxy wrapping a MockGeneDataset instance.
    Implements standard iteration generation logic to feed mini-batches into PyTorch.
    """
    def __init__(self, gene_dataset: MockGeneDataset):
        self.gene_dataset = gene_dataset
        self.X = gene_dataset.X
        self.labels = gene_dataset.labels
        self.batch_indices = gene_dataset.batch_indices
        self.n_labels = gene_dataset.n_labels
        self.n_batches = gene_dataset.n_batches
        self.nb_genes = gene_dataset.nb_genes
        self.gene_names = gene_dataset.gene_names
        self.cell_types = gene_dataset.cell_types
        self.local_means = gene_dataset.local_means
        self.local_vars = gene_dataset.local_vars
        self.norm_X = gene_dataset.norm_X
        self.corrupted_X = gene_dataset.corrupted_X
        self.protected_attributes = gene_dataset.protected_attributes
        self.dataset_versions = gene_dataset.dataset_versions
        self.gene_attribute_names = gene_dataset.gene_attribute_names
        self.cell_attribute_names = gene_dataset.cell_attribute_names
        self.cell_categorical_attribute_names = gene_dataset.cell_categorical_attribute_names
        self.attribute_mappings = gene_dataset.attribute_mappings
        self.cell_measurements_col_mappings = gene_dataset.cell_measurements_col_mappings
        
        # Self index parameters for epoch monitoring subsets
        self.indices = np.arange(self.X.shape[0])
        self.nb_cells = len(self.indices)

    def __len__(self) -> int:
        return self.X.shape[0]
        
    def __iter__(self):
        # Generator slicing yielding exact tensors expected inside CTrainer/CPosterior loops
        # Yields: sample_batch, local_l_mean, local_l_var, batch_index, labels
        batch_size = 1024
        for idx in range(0, len(self), batch_size):
            yield (
                torch.tensor(self.X[idx:idx+batch_size]),
                torch.tensor(self.local_means[idx:idx+batch_size]),
                torch.tensor(self.local_vars[idx:idx+batch_size]),
                torch.tensor(self.batch_indices[idx:idx+batch_size], dtype=torch.int64),
                torch.tensor(self.labels[idx:idx+batch_size], dtype=torch.int64)
            )

    def sequential(self, batch_size: int = 1024) -> "MockExpressionDataset":
        return self


# ==============================================================================
# 2. Existing ExpressionDataset Base Architecture (Modernized & Standalone)
# ==============================================================================

class ExpressionDataset(MockExpressionDataset): # Fixed inheritance to use Mock system

    def __init__(self, new_n_genes=None, batch_correction=False):
        self.new_n_genes = new_n_genes
        self.batch_correction = batch_correction
        self._X_dropout_removal = None
        self.cell_measurements_col_mappings = dict()

    def load_dataset_from_scVI(self, dataset: Union[MockExpressionDataset, MockGeneDataset]):
        # registers
        self.dataset_versions = dataset.dataset_versions
        self.gene_attribute_names = dataset.gene_attribute_names
        self.cell_attribute_names = dataset.cell_attribute_names
        self.cell_categorical_attribute_names = dataset.cell_categorical_attribute_names
        self.attribute_mappings = dataset.attribute_mappings
        self.cell_measurements_col_mappings = dataset.cell_measurements_col_mappings

        # initialize attributes
        self._X = dataset.X
        self._batch_indices = dataset.batch_indices
        self._labels = dataset.labels
        self.n_batches = dataset.n_batches
        self.n_labels = dataset.n_labels
        self.gene_names = dataset.gene_names
        self.cell_types = dataset.cell_types
        self.local_means = dataset.local_means
        self.local_vars = dataset.local_vars
        self._norm_X = dataset.norm_X
        self._corrupted_X = dataset.corrupted_X

        self.protected_attributes = dataset.protected_attributes

        for attr_name in self.cell_attribute_names:
            if not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))

        for attr_name in self.gene_attribute_names:
            if not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(dataset, attr_name))

    @property
    def X_dropout_removal(self) -> Union[sp_sparse.csr_matrix, np.ndarray]:
        return self._X_dropout_removal

    @X_dropout_removal.setter
    def X_dropout_removal(self, X_dropout_removal: Union[sp_sparse.csr_matrix, np.ndarray]):
        self._X_dropout_removal = X_dropout_removal

    def keep_highly_variable_genes_by_seurat(
            self,
            new_n_genes: Optional[int] = None,
            batch_correction: Optional[bool] = None
    ):
        if new_n_genes is None:
            new_n_genes = self.new_n_genes
        if batch_correction is None:
            batch_correction = self.batch_correction
            
        obs = pd.DataFrame(
            data=dict(batch=self.batch_indices.squeeze()),
            index=np.arange(self.nb_cells).astype(str)
        ).astype("category")

        counts = self.X.copy()
        if sp_sparse.issparse(counts):
            counts.data = np.round(counts.data)
        else:
            counts = np.round(counts)
            
        adata = sc.AnnData(X=counts, obs=obs)
        batch_key = "batch" if (batch_correction and self.n_batches >= 2) else None
        
        sc.pp.normalize_total(adata, target_sum=None)
        sc.pp.log1p(adata)
        
        sc.pp.highly_variable_genes(
            adata=adata,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            n_top_genes=new_n_genes,
            batch_key=batch_key,
            subset=True,
            flavor="seurat_v3"
         )
         
        genes_infos = adata.var
        subset_genes = np.array(genes_infos["highly_variable"].index, dtype=int)
        # Manually filter arrays instead of using deprecated base functions
        self.X = self.X[:, subset_genes]
        self.gene_names = self.gene_names[subset_genes]
        self.nb_genes = self.X.shape[1]

    def collate_fn_builder(
            self,
            add_attributes_and_types: Optional[Dict[str, type]] = None,
            override: bool = False,
            corrupted: bool = False,
            dropout_removal: bool = False,
    ) -> Callable[[Union[List[int], np.ndarray]], Tuple[torch.Tensor, ...]]:

        if override:
            attributes_and_types = dict()
        else:
            if dropout_removal:
                X_chosen = ("X_dropout_removal", np.float32)
            elif corrupted:
                X_chosen = ("corrupted_X", np.float32)
            else:
                X_chosen = ("X", np.float32)
            attributes_and_types = dict(
                [
                    X_chosen,
                    ("local_means", np.float32),
                    ("local_vars", np.float32),
                    ("batch_indices", np.int64),
                    ("labels", np.int64),
                ]
            )

        if add_attributes_and_types is None:
            add_attributes_and_types = dict()
        attributes_and_types.update(add_attributes_and_types)
        return partial(self.collate_fn_base, attributes_and_types)

    def make_gene_names_lower(self):
        logger.info("Making gene names lower case")
        self.gene_names = np.char.lower(self.gene_names)