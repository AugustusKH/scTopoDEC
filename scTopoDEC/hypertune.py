import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import gc
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
from .utils import set_reproducibility 

import traceback



def objective(trial, adata, args):
    """
    Optuna objective function for scTopoDEC.
    """
    gc.collect()
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
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    update_interval = trial.suggest_categorical("update_interval", [10, 20])
    soft_kmean = trial.suggest_categorical("soft_kmean", [True, False])
    tol = trial.suggest_categorical("tol", [1e-3, 1e-4, 1e-5])

    # Pretrain Params
    p_opt = trial.suggest_categorical("pre_opt", ['adam', 'rmsprop', 'adadelta'])
    p_lr = trial.suggest_float("pre_lr", 1e-3, 1e-2, log=True)

    # Clustering Params
    alpha = trial.suggest_float("alpha", 1.0, 5.0)
    n_clusters = trial.suggest_categorical("n_clusters", [10, 15, 20, 30])
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
    topo_lat_mode = trial.suggest_categorical("topo_lat_mode", ['euclid_dist', 'eff_res'])
    k = trial.suggest_categorical("k", [15, 30, 50, 100])

    # ---------- DEBUG BLOCK ----------
    print(f"\n>>> Trial {trial.number} Config:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    sys.stdout.flush() 
    # --------------------------------

    seeds = [0, 42, 123]
    trial_scores = []

    # --- MULTI-SEED LOOP ---
    for seed in seeds:
        set_reproducibility(seed=seed)
        adata_seed = adata_trial.copy()
        
        try:
            network = network_options['dec'](
                input_size=adata_seed.n_vars,
                hidden_size=hidden_size,
                activation=activation,
                noise_sd=noise_sd,
                hidden_dropout=dropout,
                n_clusters=n_clusters,
                alpha=alpha
            )
            network.build()

            train_func = ramp_train if ramp_mode else train
            
            y_pred = train_func(
                adata_seed, network, 
                epochs=args.hyperepoch,
                batch_size=batch_size,
                optimizer=optimizer,
                learning_rate=lr,
                update_interval=update_interval,
                tol=tol,
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
                verbose=False
            )

            # 3. Scoring per seed
            if args.ground_truth and args.ground_truth in adata_seed.obs:
                y_true = adata_seed.obs[args.ground_truth].values
                score = metrics.adjusted_rand_score(y_true, y_pred)
            else:
                score = network.model.history.history['loss'][-1]
            
            trial_scores.append(score)
            keras.backend.clear_session()

        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            trial_scores.append(0.0 if args.ground_truth else float('inf'))

    # Score averaging
    mean_score = np.mean(trial_scores)
    
    # Optuna minimizes: return (1 - ARI) or raw loss
    optuna_val = (1 - mean_score) if args.ground_truth else mean_score
    
    print(f"\n[SUCCESS] Trial {trial.number} Finished. Avg ARI/Loss: {mean_score:.4f}")

    return optuna_val



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
    # Initialize global reproducibility for the study itself
    set_reproducibility(seed=0)
    output_dir = os.path.join(args.outputdir, 'optuna_results')
    os.makedirs(output_dir, exist_ok=True)

    if adata_input is not None:
        adata = adata_input
    else:
        adata = io.read_dataset(args.input, 
                                transpose=args.transpose, 
                                test_split=False)

    # 1. Create and run study
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    
    print(f"Starting Optuna Optimization for {args.hypern} trials...")
    study.optimize(lambda trial: objective(trial, adata, args), n_trials=args.hypern)

    # 2. Extract best parameters
    best_params = study.best_params

    # 3. Robust check for the best model
    print("\n" + "-"*15 + " Starting Final Robustness Check " + "-"*15)
    final_verification_seeds = [50, 150, 999]
    final_ari_scores = []

    # Pre-process the data according to the best n_top_genes found
    # try:
    #     print("\n" + "-"*15 + " Starting Final Robustness Check " + "-"*15)
    #     final_verification_seeds = [50, 150, 999]
    #     final_ari_scores = []

    #     adata_hypertune = adata.copy()
    #     sc.pp.highly_variable_genes(adata_hypertune, n_top_genes=best_params['n_top_genes'], flavor='seurat_v3')
    #     hv_mask = adata_hypertune.var['highly_variable'].values
    #     adata_hypertune = adata_hypertune[:, hv_mask].copy()
    #     #adata_hypertune = adata_hypertune[:, adata_hypertune.var.highly_variable].copy()
    #     adata_hypertune = io.normalize(adata_hypertune, 
    #                                    filter_min_counts=True, 
    #                                    size_factors=True, 
    #                                    logtrans_input=True, 
    #                                    normalize_input=True)
    # except Exception:
    #     print("\n[!] CRASH DETECTED. Printing full traceback:")
    #     traceback.print_exc()
    #     sys.stdout.flush()

    adata_hypertune = adata.copy()
    adata_hypertune = io.read_dataset(adata_hypertune, check_counts=True, copy=False)
    sc.pp.highly_variable_genes(adata_hypertune, n_top_genes=best_params['n_top_genes'], flavor='seurat_v3')
    adata_hypertune = adata_hypertune[:, adata_hypertune.var.highly_variable].copy()
    adata_hypertune = io.normalize(adata_hypertune, 
                                   filter_min_counts=True, 
                                   size_factors=True, 
                                   logtrans_input=True, 
                                   normalize_input=True)

    for seed in final_verification_seeds:
        print(f">>> Verifying Seed {seed}...")
        set_reproducibility(seed) # Lock seed for this specific run
        keras.backend.clear_session()
        
        # Initialize the best model architecture
        network = network_options['dec'](
            input_size=adata_hypertune.n_vars,
            hidden_size=best_params['hidden_size'],
            activation=best_params['activation'],
            noise_sd=best_params['noise_sd'],
            hidden_dropout=best_params['dropout'],
            n_clusters=best_params['n_clusters'],
            alpha=best_params['alpha']
        )
        network.build()

        # Use ramp_train if suggested, else standard train
        train_func = ramp_train if best_params['ramp_mode'] else train
        
        y_pred = train_func(
            adata_hypertune, network, 
            epochs=args.hyperepoch,
            batch_size=best_params['batch_size'],
            optimizer=best_params['optimizer'],
            learning_rate=best_params['lr'],
            update_interval=best_params['update_interval'],
            tol=best_params['tol'],
            soft_kmean=best_params['soft_kmean'],
            loss_weights=best_params['loss_weights'],
            res_ramp=best_params['res_ramp'] if best_params['ramp_mode'] else None,
            pretrain_optimizer=best_params['pre_opt'],
            pretrain_learning_rate=best_params['pre_lr'],
            homology_dim=args.homology_dim, 
            maximum_edge_length=best_params['max_edge'],
            topo_size=best_params['topo_size'],
            order=best_params['order'],
            topo_input_mode=best_params['topo_in_mode'],
            topo_latent_mode=best_params['topo_lat_mode'],
            k=best_params['k'],
            verbose=False
        )

        if args.ground_truth is not None and args.ground_truth in adata_hypertune.obs.columns:
            y_true = adata_hypertune.obs[args.ground_truth].values
            final_ari = metrics.adjusted_rand_score(y_true, y_pred)
            final_ari_scores.append(final_ari)

    # 4. Save and print final results
    save_params = {k: (str(v) if isinstance(v, tuple) else v) for k, v in best_params.items()}
    
    # Add robustness metrics to the saved file
    if final_ari_scores:
        save_params['robustness_mean_ari'] = float(np.mean(final_ari_scores))
        save_params['robustness_std_ari'] = float(np.std(final_ari_scores))

    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump(save_params, f, indent=4)

    print("\n" + "="*40)
    print("  OPTUNA OPTIMIZATION COMPLETE  ")
    print("="*40)
   
    if final_ari_scores:
        print(f"Mean Robustness ARI: {np.mean(final_ari_scores):.4f} ± {np.std(final_ari_scores):.4f}")

    print(json.dumps(save_params, indent=4))
    
    return best_params