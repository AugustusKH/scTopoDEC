import os
import numpy as np
import pandas as pd
from pathlib import Path

def write_desc_result(data, save_dir='tmp_result', delimiter=",", cluster_key=None):
    """
    Saves DESC outputs (latent space, probabilities, and labels) to CSV.
    Updated for robust path handling and flexible cluster label selection.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Check for the embedded (latent) representation
    # DESC stores the latent space in adata.obsm['X_Embeded_z0.1'] or similar keys
    # We check for the specific 'embedded' key used in the legacy wrapper
    if 'embedded' not in data.obsm:
        raise AttributeError("The DESC latent representation ('embedded') was not found in adata.obsm.")

    # 2. Save Probability Matrix
    prob_file = save_path / 'prob_matrix.csv'
    if not prob_file.exists():
        if 'prob' in data.obsm:
            np.savetxt(prob_file, data.obsm['prob'], delimiter=delimiter)
            print(f"Saved probability matrix to {prob_file}")
        else:
            print("Skipping probabilities: 'prob' not found in adata.obsm.")
    else:
        print(f"{prob_file} already exists.")

    # 3. Save Cluster Identifications
    # Instead of iloc[:, [0, 2]], we look for the actual cluster names
    ident_file = save_path / 'cluster_ident.csv'
    if not ident_file.exists():
        # Try to find the cluster column (usually starts with 'desc_')
        if cluster_key is None:
            # Auto-detect the first column that looks like a DESC result
            desc_cols = [c for c in data.obs.columns if c.startswith('desc_')]
            cluster_key = desc_cols[0] if desc_cols else None

        if cluster_key and cluster_key in data.obs:
            # Save Barcode and Cluster ID
            out_df = pd.DataFrame(data.obs[cluster_key])
            out_df.to_csv(ident_file, sep=delimiter, header=True)
            print(f"Saved cluster labels ({cluster_key}) to {ident_file}")
        else:
            print("Skipping cluster labels: No DESC cluster column found in adata.obs.")
    else:
        print(f"{ident_file} already exists.")

    # 4. Save Embedded (Latent) Space
    embed_file = save_path / 'embedded.csv'
    if not embed_file.exists():
        np.savetxt(embed_file, data.obsm['embedded'], delimiter=delimiter)
        print(f"Saved latent embedding to {embed_file}")
    else:
        print(f"{embed_file} already exists.")

    return None