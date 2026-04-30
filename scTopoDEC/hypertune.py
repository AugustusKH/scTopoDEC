import os
import json
import optuna
import numpy as np
import scanpy as sc
import tensorflow as tf
import keras
from sklearn import metrics

from . import io
from .network import network_options
from .metric import cluster_acc
from .train import pretrain, train, ramp_train 


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 


def objective(trial, adata, args):
    """
    Optuna objective function for scTopoDEC.
    """
    keras.backend.clear_session()
    tf.config.run_functions_eagerly(False) 

    # 0. Preprocess single-cell data
    adata_trial = adata.copy()
    adata_trial = io.read_dataset(adata_trial, check_counts=True, copy=False)

    # HVG Selection 
    n_genes = trial.suggest_categorical("n_top_genes", [1000, 2000, 3000])
    
    print(f"Trial {trial.number}: Selecting top {n_genes} genes...")
    sc.pp.highly_variable_genes(adata_trial, n_top_genes=n_genes, flavor='seurat_v3')
    adata_trial = adata_trial[:, adata_trial.var.highly_variable].copy()

    # Normalization and scaling 
    adata_trial = io.normalize(adata_trial, 
                               filter_min_counts=True,
                               size_factors=True, 
                               logtrans_input=True, 
                               normalize_input=True)

    # 1. Define Search Space Dynamically
    # Model Params
    hidden_choices = [
        (256, 128, 64, 128, 256), (256, 128, 64, 32, 64, 128, 256), (256, 128, 64, 32, 16, 32, 64, 128, 256),
        (256, 128, 256), (256, 64, 256), (128, 64, 32, 64, 128), (128, 64, 128), (128, 32, 128), 
        (256, 64, 128, 64, 256), (256, 128, 64, 128, 64, 128, 256), (256, 128, 64, 32, 64, 32, 64, 128, 256),
        (256, 128, 64, 32, 64, 128, 64, 32, 64, 128, 256), (256, 128, 128, 128, 256), (128, 64, 32, 64, 32, 64, 128)
    ]
    hidden_size = trial.suggest_categorical("hidden_size", hidden_choices)
    activation = trial.suggest_categorical("activation", ['relu', 'LeakyReLU'])
    noise_sd = trial.suggest_float("noise_sd", 0.0, 0.5)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    # Training Params
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    optimizer = trial.suggest_categorical("optimizer", ['adam', 'rmsprop', 'adadelta'])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    update_interval = trial.suggest_categorical("update_interval", [5, 10, 20])
    soft_kmean = trial.suggest_categorical("soft_kmean", [True, False])

    # Pretrain Params
    p_opt = trial.suggest_categorical("pre_opt", ['adam', 'rmsprop', 'adadelta'])
    p_lr = trial.suggest_float("pre_lr", 1e-3, 1e-1, log=True)

    # Clustering Params
    alpha = trial.suggest_float("alpha", 1.0, 5.0)
    n_clusters = trial.suggest_categorical("n_clusters", [5, 10, 15, 20, 30])
    weight_choices = [
        (1.0, 1.0, 0.0, 0.1), (1.0, 1.0, 0.0, 0.5), (1.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, 2.0), (1.0, 0.5, 0.0, 1.0), (1.0, 1.0, 0.1, 0.5),
        (0.5, 1.0, 0.0, 1.0), (0.5, 1.0, 0.0, 0.5), (0.5, 0.5, 0.0, 1.0)
    ]
    loss_weights = trial.suggest_categorical("loss_weights", weight_choices)
    ramp_mode = trial.suggest_categorical("ramp_mode", [True, False])
    res_ramp = trial.suggest_categorical("res_ramp", [(0.0, 0.1, 0.2, 0.5, 1.0), (0.1, 0.5, 1.0), (0.5, 1.0)])

    # Topology Params
    max_edge = trial.suggest_float("max_edge", 0.5, 2.0)
    topo_size = trial.suggest_categorical("topo_size", [45, 64, 128])
    order = trial.suggest_float("order", 0.5, 2.0)
    topo_in_mode = trial.suggest_categorical("topo_in_mode", ['umap', 'pca_dist', 'knn', 'eff_res'])
    topo_lat_mode = trial.suggest_categorical("topo_lat_mode", ['raw', 'inner_product', 'euclid_dist', 'eff_res'])
    k = trial.suggest_categorical("k", [15, 30, 50, 100])

    try:
        # 2. Initialize Network
        network = network_options['dec'](
            input_size=adata_trial.n_vars,
            hidden_size=hidden_size,
            activation=activation,
            noise_sd=noise_sd,
            hidden_dropout=dropout,
            n_clusters=n_clusters,
            alpha=alpha
        )
        network.build()

        # This will stop training the moment loss becomes NaN
        nan_callback = keras.callbacks.TerminateOnNaN()

        train_func = ramp_train if ramp_mode else train
        
        y_pred = train_func(
            adata_trial, network, 
            epochs=args.hyperepoch,
            batch_size=batch_size,
            optimizer=optimizer,
            learning_rate=lr,
            update_interval=update_interval,
            soft_kmean=soft_kmean,
            loss_weights=loss_weights,
            res_ramp=res_ramp if ramp_mode else None,
            pretrain_optimizer=p_opt,
            pretrain_learning_rate=p_lr,
            homology_dim=args.homology_dim, 
            maximum_edge_length=max_edge,
            topo_size=topo_size,
            order=order,
            topo_input_mode=topo_in_mode,
            topo_latent_mode=topo_lat_mode,
            k=k,
            t=args.t, 
            callbacks=[nan_callback],
            verbose=False
        )

        # 3. Skip condition if the model stopped due to NaN
        if getattr(network.model, 'stop_training', False):
            print(f"Trial {trial.number} encountered NaN and was skipped.")
            raise optuna.exceptions.TrialPruned()

        # Evaluate the model on a small sample to check weights/outputs
        eval_loss = network.model.evaluate([adata_trial.X[:5], adata_trial.obs['size_factors'][:5]], 
                                            adata_trial.X[:5], verbose=0)
        if np.isnan(eval_loss).any():
             raise optuna.exceptions.TrialPruned()
        
        # 4. Scoring
        if args.ground_truth and args.ground_truth in adata_trial.obs:
            y_true = adata_trial.obs[args.ground_truth].values
            # Optuna minimizes by default, so we use (1 - ARI)
            score = 1 - metrics.adjusted_rand_score(y_true, y_pred)
        else:
            score = network.model.history.history['loss'][-1]

        # Check NaN on the score
        if np.isnan(score):
            raise optuna.exceptions.TrialPruned()
        
        return score

    except optuna.exceptions.TrialPruned:
        raise # Re-raise so Optuna marks it as Pruned

    except Exception as e:
        print(f"\n[!] Trial {trial.number} failed: {e}")
        return float('inf') # Penalty for crashes

def hyperparams_tune(args, adata_input=None):
    """
    Hyperparameter tuning using Optuna optimization.
    This function automates the search for optimal hyperparameters by executing 
    multiple trials of the scTopoDEC model. It manages data loading, 
    initializes the Optuna study, and saves the best-performing configuration.

    Args:
        args (SimpleNamespace): A configuration object containing:
            - outputdir (str): Path to save optimization results and logs.
            - input (str): Path to the single-cell dataset (used if adata_input is None).
            - transpose (bool): Whether to transpose the input matrix (genes vs. cells).
            - hypern (int): The total number of optimization trials to execute.
            - hyperepoch (int): Number of training epochs per individual trial.
            - ground_truth (str): Metadata column in adata.obs used for scoring (ARI/AMI).
            - homology_dim (int): Topological dimension (0 for components, 1 for loops).
        adata_input (AnnData, optional): An existing AnnData object. If provided, 
            skips disk I/O and uses this object directly for optimization. 
            Defaults to None.

    Returns:
        dict: A dictionary containing the hyperparameters from the best-performing trial.
    """
    keras.utils.set_random_seed(42)
    output_dir = os.path.join(args.outputdir, 'optuna_results')
    os.makedirs(output_dir, exist_ok=True)

    if adata_input is not None:
        adata = adata_input
    else:
        adata = io.read_dataset(args.input, 
                                transpose=args.transpose, 
                                test_split=False)

    # 1. Create and run study
    # Use MedianPruner to kill poorly performing trials early
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    print(f"Starting Optuna Optimization for {args.hypern} trials...")
    study.optimize(lambda trial: objective(trial, adata, args), n_trials=args.hypern)

    # 2. Save results
    best_params = study.best_params
    # Convert tuples to strings for JSON serializability
    save_params = {k: (str(v) if isinstance(v, tuple) else v) for k, v in best_params.items()}

    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump(save_params, f, indent=4)

    print("\n" + "="*40)
    print("  OPTUNA OPTIMIZATION COMPLETE  ")
    print("="*40)
    print(f"Best Loss/ARI: {study.best_value}")
    print(json.dumps(save_params, indent=4))
    
    return best_params