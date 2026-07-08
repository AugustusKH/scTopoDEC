import argparse
import scanpy as sc
import sys
import ast
import gc
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from .api import scTopoDEC

def main():
    parser = argparse.ArgumentParser(description='scTopoDEC: Topological Deep Embedded Clustering')
    
    # 1. Input/Output
    parser.add_argument('input', type=str, help='Path to input .h5ad file')
    parser.add_argument('--output', type=str, default='scTopoDEC_results.h5ad', help='Output filename')

    # 2. Load and save weights
    parser.add_argument('--train_output_dir', type=str, default=None, 
                        help='Directory to save final training weights and metadata')
    parser.add_argument('--pretrain_output_dir', type=str, default=None, 
                        help='Directory to save pretraining checkpoints')
    parser.add_argument('--initial_pretrain_weights', type=str, default=None, 
                        help='Path to existing .weights.h5 file to skip pretraining')
    parser.add_argument('--initial_train_weights', type=str, default=None, 
                        help='Path to existing .weights.h5 file to resume clustering')
    parser.add_argument('--save_pretrain_weights', action='store_true', help='Enable pretrain saving')
    parser.add_argument('--save_train_weights', action='store_true', help='Enable final weight saving')

    # 3. Network architecture args
    parser.add_argument('--ae_type', type=str, default='dec', choices=['ae', 'zinb', 'dec'], help='Model type')
    parser.add_argument('--mode', type=str, default='clustering', choices=['clustering', 'denoise', 'latent', 'full'], 
                        help='Mode type')
    parser.add_argument('--hidden_size', type=str, default="(256, 32, 256)", 
                        help='Network architecture as a tuple string')
    parser.add_argument('--loss_weights', type=str, default="(1.0, 10.0, 0.1, 1.0)", 
                        help='Loss weights (ZINB, Cluster, SoftK, Topo) as a tuple string')
    parser.add_argument('--batchnorm', type=bool, default=True, help='Use Batch Normalization')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--init', type=str, default='glorot_uniform', help='Weight initialization method')

    # 4. Clustering and training args
    parser.add_argument('--n_clusters', type=int, default=0, 
                        help='Number of clusters, if set to 0, will use Leiden to determine optimal number of clusters')
    parser.add_argument('--ramp_mode', action='store_true', help='Enable iterative resolution ramping')
    parser.add_argument('--res_ramp', type=str, default="(0.0, 0.1, 0.2, 0.5, 1.0)", help='Ramping factors')
    parser.add_argument('--epochs', type=int, default=500, help='Max training epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=800, help='Autoencoder pretraining epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--pretrain_lr', type=float, default=0.001, help='Pretraining learning rate')
    parser.add_argument('--update_interval', type=int, default=10, help='Epochs between target distribution updates')
    parser.add_argument('--tol', type=float, default=1e-3, help='Convergence tolerance')
    parser.add_argument('--no_cluster_stop', action='store_false', dest='cluster_early_stop', 
                        help='Disable early stopping during the clustering/DEC phase')
    parser.set_defaults(cluster_early_stop=True)
    parser.add_argument('--cluster_patience', type=int, default=15, 
                        help='Patience for clustering early stopping')

    # 5. Topology specific args 
    parser.add_argument('--homology_dim', type=int, default=1, choices=[0, 1, 2], 
                        help='Dimension of persistent homology (1 for loops/trajectories)')
    parser.add_argument('--maximum_edge_length', type=float, default=1.0, 
                        help='Filtration cutoff (prevents OOM on large datasets)')
    parser.add_argument('--topo_size', type=int, default=256, 
                        help='Number of cells sampled for topological loss calculation')
    parser.add_argument('--pg_dist', type=str, default='wd', choices=['wd', 'mse', 'weight_mse'], 
                        help='Distance metric for persistent diagrams')
    parser.add_argument('--order', type=float, default=1.0, 
                        help='Wasserstein exponent q for diagram distance')
    parser.add_argument('--topo_input_mode', type=str, default='eff_res', 
                        help='Ground-truth manifold mode (pca, umap, pca_dist, umap_dist, knn, eff_res, diffusion)')
    parser.add_argument('--topo_latent_mode', type=str, default='euclid_dist', 
                        help='Latent space representation mode (raw, inner_product, euclid_dist, knn, eff_res, diffusion)')
    parser.add_argument('--n_components', type=int, default=50, 
                        help='Dimensions to retain for topological ground-truth')
    parser.add_argument('--k', type=int, default=100, 
                        help='Nearest neighbors for graph construction')
    parser.add_argument('--t', type=int, default=8, 
                        help='Diffusion time for transition matrix iterations')
    
    # 6. Data processing args
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
    
    # 7. Regularization and noise 
    parser.add_argument('--noise_sd', type=float, default=0.4, 
                        help='Standard deviation of Gaussian noise added to input')
    parser.add_argument('--hidden_dropout', type=float, default=0.05, 
                        help='Dropout rate applied to hidden layers (0.0 to 1.0)')
    parser.add_argument('--alpha', type=float, default=2.0, 
                        help='Degrees of freedom for Student’s t-distribution in clustering')
    parser.add_argument('--gamma', type=float, default=1.0, 
                        help='The frequency smoothing exponent used for the target distribution p')

    # 8. Preprocessing and normalization
    parser.add_argument('--no_norm', action='store_false', dest='normalize_per_cell', 
                        help='Disable library size normalization')
    parser.set_defaults(normalize_per_cell=True)
    parser.add_argument('--no_scale', action='store_false', dest='scale', 
                        help='Disable input scaling')
    parser.set_defaults(scale=True)
    parser.add_argument('--no_log', action='store_false', dest='log1p', 
                        help='Disable log(1+x) transformation')
    parser.set_defaults(log1p=True)

    # 9. Optimizer and stability (DEC/pretrain tuning)
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for DEC')
    parser.add_argument('--pretrain_optimizer', type=str, default='adam', help='Optimizer for Pretraining')
    parser.add_argument('--no_soft_kmean', action='store_false', dest='soft_kmean',
                        help='Disable soft k-mean loss (only relevant for DEC)')
    parser.add_argument('--cluster_early_stop', action='store_true', 
                        help='Enable early stopping during the clustering/DEC phase')
    parser.add_argument('--reduce_lr', type=int, default=20, 
                        help='Patience for Reducing Learning Rate on plateau')
    parser.add_argument('--early_stop', type=int, default=30, 
                        help='Patience for Early Stopping based on validation loss')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')

    args = parser.parse_args()

    # --- Data loading ---
    try:
        print(f"Loading data: {args.input}")
        adata = sc.read(args.input)
        if args.transpose:
            print("Transposing input matrix...")
            adata = adata.T
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # --- Parameter conversion ---
    try:
        hidden_size_obj = ast.literal_eval(args.hidden_size)
        loss_weights_obj = ast.literal_eval(args.loss_weights)
        res_ramp_obj = ast.literal_eval(args.res_ramp)
    except Exception as e:
        print(f"Error parsing hidden_size or loss_weights: {e}")
        sys.exit(1)

    # --- Run API ---
    if args.hyper:
        print(">>> Entering Hyperparameter Optimization Mode")
        
        # 1. Preprocess data strictly for Hyperopt evaluation
        print("Preprocessing data for hyperparameter tuning...")
        sc.pp.filter_genes(adata, min_cells=5)
        sc.pp.filter_cells(adata, min_counts=5)
        adata.layers["counts"] = adata.X.copy() # Save raw count data
        
        # 2. Encode ground truth labels for scoring
        if args.ground_truth is not None and args.ground_truth in adata.obs.columns:
            le = LabelEncoder()
            adata.obs['ground_truth_label'] = adata.obs[args.ground_truth].copy()
            adata.obs['ground_truth'] = le.fit_transform(adata.obs[args.ground_truth])
        else:
            print(f"[!] Warning: Ground truth '{args.ground_truth}' not found. Optimizing based on internal loss.")
            
        # Clean memory before launching Optuna
        gc.collect()
        tf.keras.backend.clear_session()
        
        # 3. Launch hyperparams_tune using the correct local import path
        from .hypertune import hyperparams_tune
        hyperparams_tune(args, adata_input=adata)
        
        print(f"Optimization finished. Results saved to {args.outputdir}/optuna_results/")
        sys.exit(0) # Exit the script once tuning is finished

    else:
        print(">>> Entering Standard Training Mode")
        scTopoDEC(
            adata,
            ae_type=args.ae_type,
            mode=args.mode,
            n_clusters=args.n_clusters,
            alpha=args.alpha,
            gamma=args.gamma,
            hidden_size=hidden_size_obj,
            loss_weights=loss_weights_obj,
            noise_sd=args.noise_sd,
            hidden_dropout=args.hidden_dropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
            init=args.init,
            soft_kmean=args.soft_kmean,
            use_hvg=args.use_hvg,
            n_top_genes=args.n_top_genes,
            normalize_per_cell=args.normalize_per_cell,
            scale=args.scale,
            log1p=args.log1p,
            epochs=args.epochs,
            optimizer=args.optimizer,
            learning_rate=args.lr,
            update_interval=args.update_interval,
            tol=args.tol,
            cluster_early_stop=args.cluster_early_stop, 
            early_stop=args.cluster_patience,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_optimizer=args.pretrain_optimizer,
            pretrain_learning_rate=args.pretrain_lr,
            reduce_lr=args.reduce_lr,
            batch_size=args.batch_size,
            ground_truth=args.ground_truth,
            ramp_mode=args.ramp_mode,
            res_ramp=res_ramp_obj,
            homology_dim=args.homology_dim,
            maximum_edge_length=args.maximum_edge_length,
            topo_size=args.topo_size,
            pg_dist=args.pg_dist,
            order=args.order,
            topo_input_mode=args.topo_input_mode,
            topo_latent_mode=args.topo_latent_mode,
            n_components=args.n_components,
            k=args.k,
            t=args.t,
            train_output_dir=args.train_output_dir,
            pretrain_output_dir=args.pretrain_output_dir,
            initial_pretrain_weights=args.initial_pretrain_weights,
            initial_train_weights=args.initial_train_weights,
            save_pretrain_weights=args.save_pretrain_weights,
            save_train_weights=args.save_train_weights,
            copy=False,
            verbose=True
        )

        # --- Result Saving ---
        print(f"Saving results to {args.output}")
        adata.write(args.output)
        print("Done!")

if __name__ == "__main__":
    main()