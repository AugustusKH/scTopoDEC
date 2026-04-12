import argparse
import scanpy as sc
import sys
import ast
from .api import scTopoDEC

def main():
    parser = argparse.ArgumentParser(description='scTopoDEC: Topological Deep Embedded Clustering')
    
    # 1. Positional args (input files)
    parser.add_argument('input', type=str, help='Path to input .h5ad file')

    # 2. Network achitecture args
    # We use 'ast.literal_eval' to turn string "(256, 64)" into a real tuple
    parser.add_argument('--hidden_size', type=str, default="(256, 64, 32, 64, 256)", 
                        help='Network architecture as a tuple string')
    parser.add_argument('--loss_weights', type=str, default="[1, 1, 0]", 
                        help='Loss weights [ZINB, Cluster, Topo] as a list string')
    parser.add_argument('--batchnorm', type=bool, default=True, help='Use Batch Normalization')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')

    # 3. Clustering and training args
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters')
    parser.add_argument('--ramp_mode', action='store_true', help='Enable iterative resolution ramping')
    parser.add_argument('--epochs', type=int, default=300, help='Max training epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='Autoencoder pretraining epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--pretrain_lr', type=float, default=0.01, help='Pretraining learning rate')
    
    # 4. Data Processing Args
    parser.add_argument('--n_top_genes', type=int, default=2000, help='Number of HVGs to use')
    parser.add_argument('--output', type=str, default='scTopoDEC_results.h5ad', help='Output filename')

    args = parser.parse_args()

    # --- Data Loading ---
    try:
        print(f"Loading data: {args.input}")
        adata = sc.read(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # --- Parameter Conversion ---
    # This converts the terminal strings back into Python objects
    try:
        hidden_size_obj = ast.literal_eval(args.hidden_size)
        loss_weights_obj = ast.literal_eval(args.loss_weights)
    except Exception as e:
        print(f"Error parsing hidden_size or loss_weights: {e}")
        sys.exit(1)

    # --- Run API (scTopoDEC) ---
    # We map the CLI args directly to your scTopoDEC function parameters
    scTopoDEC(
        adata,
        n_clusters=args.n_clusters,
        n_top_genes=args.n_top_genes,
        hidden_size=hidden_size_obj,
        loss_weights=loss_weights_obj,
        batchnorm=args.batchnorm,
        activation=args.activation,
        epochs=args.epochs,
        learning_rate=args.lr,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_learning_rate=args.pretrain_lr,
        ramp_mode=args.ramp_mode,
        copy=False,         # Modify in-place for memory efficiency
        verbose=True        # Good for terminal users
    )

    # --- Result Saving ---
    print(f"Saving results to {args.output}")
    adata.write(args.output)
    print("Done!")

if __name__ == "__main__":
    main()