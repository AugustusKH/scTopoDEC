# run_hyper.py
import sys
import os
import gc
import tensorflow as tf
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from types import SimpleNamespace
from scTopoDEC.hypertune import hyperparams_tune

def load_general_data(input_path):
    """
    Generalized loader for different single-cell file formats.
    """
    extension = os.path.splitext(input_path)[1].lower()
    print(f"Loading dataset from: {input_path}")
    
    if extension == '.h5ad':
        return sc.read_h5ad(input_path)
    elif extension == '.csv':
        return sc.read_csv(input_path).transpose()
    elif extension in ['.mtx', '.gz']:
        return sc.read_10x_mtx(os.path.dirname(input_path))
    else:
        raise ValueError(f"Unsupported file format: {extension}")
    
def preprocess_data(adata, ground_truth=None):
    """
    Preprocess adata and encode ground truth label
    """
    # Preprocessing
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_cells(adata, min_counts=5)
    adata.layers["counts"] = adata.X.copy() # Saving count data

    # Ground truth encoding
    if ground_truth is not None and ground_truth in adata.obs:
        le = LabelEncoder()
        adata.obs['ground_truth_label'] = adata.obs['ground_truth'].copy()
        adata.obs['ground_truth'] = le.fit_transform(adata.obs[ground_truth])

    return adata

def main():
    # 1. Setup Arguments
    args = SimpleNamespace(
        input="sc_data.h5ad",                   # Path to your specific dataset
        outputdir="./results",
        transpose=False,
        hypern=30,                              # Number of Optuna trials
        hyperepoch=50,                          # Epochs per trial
        ground_truth="ground_truth_label",      # Change this to the column name in your adata.obs
        t=8,                                    # Diffusion time
        homology_dim=1                          # Persistence dimension
    )

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    try:
        # 2. Load and preprocess the data
        adata = load_general_data(args.input)
        preprocess_data(adata, ground_truth=args.ground_truth)
        
        # Data Integrity Check
        if args.ground_truth not in adata.obs.columns:
            available = list(adata.obs.columns)
            print(f"[!] Warning: '{args.ground_truth}' not found in metadata.")
            print(f"Available columns: {available}")
            # Optional: fall back to the first available column or index
            # args.ground_truth = available[0] 

        # Clear any leftover memory from preprocessing
        gc.collect()
        tf.keras.backend.clear_session()

        print(f"Starting Optuna Study on {adata.n_obs} cells and {adata.n_vars} genes...")
        
        # 3. Pass both the args and the loaded adata object
        best_config = hyperparams_tune(args, adata_input=adata)
        
        print("\n" + "="*30)
        print("Optimization Finished Successfully.")
        print(f"Best Configuration: {args.outputdir}/optuna_results/best_config.json")
        print("="*30)
        
    except KeyboardInterrupt:
        print("\n[!] Optimization interrupted by user.")
    except Exception as e:
        print(f"\n[!] An error occurred during the study: {e}")

if __name__ == "__main__":
    main()