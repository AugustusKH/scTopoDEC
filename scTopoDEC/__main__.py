import argparse
import scanpy as sc
import sys
import ast
import tensorflow as tf
from .api import scTopoDEC

def main():
    parser = argparse.ArgumentParser(description='scTopoDEC: Topological Deep Embedded Clustering')
    
    # 1. Input/Output
    parser.add_argument('input', type=str, help='Path to input .h5ad file')
    parser.add_argument('--output', type=str, default='scTopoDEC_results.h5ad', help='Output filename')

    # 2. Network achitecture args
    parser.add_argument('--ae_type', type=str, default='dec', choices=['ae', 'zinb', 'dec'], help='Model type')
    parser.add_argument('--hidden_size', type=str, default="(256, 64, 32, 64, 256)", 
                        help='Network architecture as a tuple string')
    parser.add_argument('--loss_weights', type=str, default="(1, 1, 0, 1.0)", 
                        help='Loss weights (ZINB, Cluster, SoftK, Topo) as a tuple string')
    parser.add_argument('--batchnorm', type=bool, default=True, help='Use Batch Normalization')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')

    # 3. Clustering and training args
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters')
    parser.add_argument('--ramp_mode', action='store_true', help='Enable iterative resolution ramping')
    parser.add_argument('--res_ramp', type=str, default="(0.1, 0.5, 1.0)", help='Ramping factors')
    parser.add_argument('--epochs', type=int, default=300, help='Max training epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='Autoencoder pretraining epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--pretrain_lr', type=float, default=0.01, help='Pretraining learning rate')
    parser.add_argument('--update_interval', type=int, default=10, help='Epochs between target distribution updates')
    parser.add_argument('--tol', type=float, default=1e-3, help='Convergence tolerance')

    # 4. Topology specific args 
    parser.add_argument('--homology_dim', type=int, default=1, choices=[0, 1, 2], 
                        help='Dimension of persistent homology (1 for loops/trajectories)')
    parser.add_argument('--maximum_edge_length', type=float, default=2.0, 
                        help='Filtration cutoff (prevents OOM on large datasets)')
    
    # 5. Data processing args
    parser.add_argument('--n_top_genes', type=int, default=2000, help='Number of HVGs to use')
    parser.add_argument('--no_hvg', action='store_false', dest='use_hvg', help='Disable HVG selection')
    parser.set_defaults(use_hvg=True)
    parser.add_argument('--ground_truth', type=str, default=None, help='adata.obs key for true labels')

    # 6. Hyperparameter optimization args 
    parser.add_argument('--hyper', action='store_true', help='Run Hyperparameter Optimization mode')
    parser.add_argument('--hypern', type=int, default=50, 
                        help='Number of trials for Hyperopt')
    parser.add_argument('--hyperepoch', type=int, default=100, 
                        help='Number of epochs per trial during optimization')
    parser.add_argument('--outputdir', type=str, default='./', 
                        help='Directory to save hyperopt results')
    parser.add_argument('--transpose', action='store_true', 
                        help='Transpose input matrix (cells as columns)')

    args = parser.parse_args()

    # --- Data loading ---
    try:
        print(f"Loading data: {args.input}")
        adata = sc.read(args.input)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # --- Parameter conversion ---
    # This converts the terminal strings back into Python objects
    try:
        hidden_size_obj = ast.literal_eval(args.hidden_size)
        loss_weights_obj = ast.literal_eval(args.loss_weights)
        res_ramp_obj = ast.literal_eval(args.res_ramp)
    except Exception as e:
        print(f"Error parsing hidden_size or loss_weights: {e}")
        sys.exit(1)

    # --- Run API (scTopoDEC) ---
    # We map the CLI args directly to your scTopoDEC function parameters
    if args.hyper:
        print(">>> Entering Hyperparameter Optimization Mode")
        from .hyper import hyper
        hyper(args) 
    else:
        print(">>> Entering Standard Training Mode")
        scTopoDEC(
            adata,
            ae_type=args.ae_type,
            n_clusters=args.n_clusters,
            use_hvg=args.use_hvg,
            n_top_genes=args.n_top_genes,
            hidden_size=hidden_size_obj,
            loss_weights=loss_weights_obj,
            batchnorm=args.batchnorm,
            activation=args.activation,
            epochs=args.epochs,
            learning_rate=args.lr,
            update_interval=args.update_interval,
            tol=args.tol,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_learning_rate=args.pretrain_lr,
            ground_truth=args.ground_truth,
            ramp_mode=args.ramp_mode,
            res_ramp=res_ramp_obj,
            homology_dim=args.homology_dim,
            maximum_edge_length=args.maximum_edge_length,
            copy=False,
            verbose=True
        )

    # --- Result Saving ---
    print(f"Saving results to {args.output}")
    adata.write(args.output)
    print("Done!")

if __name__ == "__main__":
    main()