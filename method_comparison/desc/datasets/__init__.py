import os
from pathlib import Path
from anndata import read_h5ad

# Get the directory where this script is located
DATA_DIR = Path(__file__).parent

def pbmc():
    """Load the raw PBMC dataset."""
    file_path = DATA_DIR / 'pbmc.h5ad'
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return read_h5ad(file_path)

def pbmc_processed():
    """Load the preprocessed PBMC dataset."""
    file_path = DATA_DIR / 'pbmc_processed.h5ad'
    if not file_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {file_path}")
    return read_h5ad(file_path)

def get_pbmc(save_dir='tmp_data'):
    """
    Copies the PBMC dataset to a target directory.
    Created for DESC/scTopoDEC compatibility tests.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    target_file = save_path / 'pbmc.h5ad'
    
    if not target_file.exists():
        print(f"Transferring pbmc.h5ad to {save_path}...")
        adata = pbmc()
        adata.write(target_file)
        print('Success: pbmc.h5ad is ready.')
    else:
        print(f"The pbmc data already exists in {save_dir}.")
    
    return None