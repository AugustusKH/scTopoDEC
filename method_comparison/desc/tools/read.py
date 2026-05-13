import pandas as pd
import scanpy as sc
from anndata import AnnData
from pathlib import Path

def read_10X(data_path, var_names='gene_symbols'):
    """
    Modernized reader for 10X Genomics MTX folders.
    Supports both legacy (genes.tsv) and modern (features.tsv.gz) formats.
    """
    path = Path(data_path)
    
    # 1. Handle Matrix file (support .gz if present)
    mtx_file = path / 'matrix.mtx'
    if not mtx_file.exists():
        mtx_file = path / 'matrix.mtx.gz'
    
    # scanpy's read_mtx is generally more robust for modern anndata
    adata = sc.read_mtx(mtx_file).T

    # 2. Handle Genes/Features file
    # Modern 10X uses 'features.tsv.gz', legacy uses 'genes.tsv'
    genes_file = path / 'features.tsv.gz'
    if not genes_file.exists():
        genes_file = path / 'genes.tsv'
        if not genes_file.exists():
            genes_file = path / 'genes.tsv.gz'

    # Read the feature metadata
    genes = pd.read_csv(genes_file, header=None, sep='\t')
    
    # Assign metadata columns
    # 10X typically has: [ID, Symbol, Type]
    adata.var['gene_ids'] = genes[0].values
    adata.var['gene_symbols'] = genes[1].values

    # 3. Handle Variable Names (Gene IDs or Symbols)
    if var_names not in ['gene_symbols', 'gene_ids']:
        raise ValueError('var_names must be "gene_symbols" or "gene_ids"')

    target_names = genes[1] if var_names == 'gene_symbols' else genes[0]

    # Ensure uniqueness using modern AnnData method
    adata.var_names = target_names.astype(str)
    if not adata.var_names.is_unique:
        print('var_names are not unique, applying make_index_unique')
        adata.var_names_make_unique()

    # 4. Handle Barcodes/Cells
    barcodes_file = path / 'barcodes.tsv.gz'
    if not barcodes_file.exists():
        barcodes_file = path / 'barcodes.tsv'
        
    cells = pd.read_csv(barcodes_file, header=None, sep='\t')
    adata.obs_names = cells[0].astype(str)
    adata.obs['barcode'] = cells[0].values
    
    return adata