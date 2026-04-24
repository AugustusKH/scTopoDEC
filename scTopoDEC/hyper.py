import os
import pickle
import json
import numpy as np
import tensorflow as tf
import keras
from keras import optimizers
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn import metrics

from . import io
from .network import network_options
from .metric import cluster_acc
from .train import pretrain, train, ramp_train 

def hyper(args):
    """
    Bayesian Hyperparameter Optimization for scTopoDEC.
    Optimizes for ZINB reconstruction, Clustering, and Topology preservation.
    """
    # 1. Reproducibility setup
    keras.utils.set_random_seed(42)
    
    output_dir = os.path.join(args.outputdir, 'hyperopt_results')
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load dataset
    adata = io.read_dataset(args.input, 
                            transpose=args.transpose, 
                            test_split=False)

    # 3. Define search space choices
    hidden_choices = [
        (512, 256, 256, 256, 512),
        (512, 256, 128, 256, 512),
        (512, 128, 256, 128, 512),
        (512, 256, 128, 64, 128, 256, 512),
        (512, 256, 128, 64, 32, 64, 128, 256, 512),
        (256, 128, 128, 128, 256), 
        (256, 128, 64, 128, 256), 
        (256, 128, 64, 32, 64, 128, 256), 
        (256, 128, 32, 128, 256), 
        (256, 64, 128, 64, 256),
        (256, 64, 32, 64, 256), 
        (128, 64, 32, 64, 128),
        (256, 64, 256),
        (256, 32, 256),
        (128, 64, 128),
        (128, 32, 128),
        (64, 32, 64)
    ]
    act_choices = ['relu', 'LeakyReLU']
    opt_choices = ['adam', 'rmsprop', 'adadelta']
    ramp_choices = [(0.0, 0.1, 0.2, 0.5, 1.0), (0.1, 0.5, 1.0), (0.5, 1.0)]

    # 4-weight tuple: (ZINB, KL, SoftK, Topo)
    weight_choices = [
        (1.0, 1.0, 0.0, 0.1),  # Low Topo
        (1.0, 1.0, 0.0, 0.5),  # Medium Topo
        (1.0, 1.0, 0.0, 1.0),  # Equal Balance
        (1.0, 1.0, 0.0, 2.0),  # High Topo pressure
        (1.0, 0.5, 0.0, 1.0),  # Topo over Clustering
        (0.5, 1.0, 0.0, 0.5),  # Cluster over Reconstruction
        (1.0, 1.0, 0.1, 0.5),  # Mixed Kmean 
        (1.0, 1.0, 0.1, 1.0)   # Balanced Mixed
    ]

    # 5. Topological representation modes
    topo_input_choices = ['umap', 'pca_dist', 'umap_dist', 'knn', 'eff_res']
    topo_latent_choices = ['raw', 'inner_product', 'euclid_dist', 'eff_res']

    hyper_params = {
        "model": {
            "hidden_size": hp.choice("m_hidden", hidden_choices),
            "activation": hp.choice("m_act", act_choices),
            "noise_sd": hp.uniform("m_noise", 0.0, 0.5),
            "dropout": hp.uniform("m_do", 0.0, 0.5),
        },
        "training": {
            "batch_size": hp.choice("t_batch", (64, 128, 256, 512)),
            "optimizer": hp.choice("t_opt", opt_choices),
            "lr": hp.loguniform("t_lr", np.log(1e-4), np.log(1e-2)),
            "update_interval": hp.choice("t_up_int", (5, 10, 20)),
            "soft_kmean": hp.choice("t_softk", (True, False)),
        },
        "pretrain": {
            "optimizer": hp.choice("p_opt", opt_choices),
            "lr": hp.loguniform("p_lr", np.log(1e-3), np.log(1e-1)),
        },
        "clustering": {
            "n_clusters": hp.choice("c_num", (5, 10, 15, 20, 30)),
            "loss_weights": hp.choice("c_weights", weight_choices),
            "ramp_mode": hp.choice("c_ramp_m", (True, False)),
            "res_ramp": hp.choice("c_ramp_v", ramp_choices),
        },
        "topology": {
            "max_edge": hp.uniform("t_max_edge", 0.5, 2.0),
            "topo_size": hp.choice("t_size", (45, 64, 128)),
            "topo_input_mode": hp.choice("t_in_mode", topo_input_choices),
            "topo_latent_mode": hp.choice("t_lat_mode", topo_latent_choices),
            "k": hp.choice("t_k", (15, 30))
        },
        "fit": {
            "epochs": args.hyperepoch
        }
    }

    # 4. Objective Function (The Trial Runner)
    def objective(params):
        keras.backend.clear_session()
        # GUDHI requires eager mode; ensure graph mode isn't forced by a hidden @tf.function
        tf.config.run_functions_eagerly(True) 
        
        m, t, c, p, f = params['model'], params['topology'], params['clustering'], params['pretrain'], params['training']

        try:
            # 1. Initialize Network
            network = network_options['dec'](
                input_size=adata.n_vars,
                hidden_size=m['hidden_size'],
                activation=m['activation'],
                noise_sd=m['noise_sd'],
                hidden_dropout=m['dropout'],
                n_clusters=c['n_clusters']
            )
            network.build()

            train_func = ramp_train if c['ramp_mode'] else train
            
            # Ensure all keys exist in params before calling
            y_pred = train_func(
                adata, 
                network, 
                epochs=f['epochs'],
                batch_size=f['batch_size'],
                optimizer=f['optimizer'],
                learning_rate=f['lr'],
                update_interval=f['update_interval'],
                soft_kmean=f['soft_kmean'],
                loss_weights=c['loss_weights'],
                res_ramp=c['res_ramp'] if c['ramp_mode'] else None,
                # Pretrain args
                pretrain_optimizer=p['optimizer'],
                pretrain_learning_rate=p['lr'],
                # Topo args (Corrected keys to match dictionary)
                homology_dim=args.homology_dim, 
                maximum_edge_length=t['max_edge'],
                topo_size=t['topo_size'],
                topo_input_mode=t['topo_input_mode'],
                topo_latent_mode=t['topo_latent_mode'],
                k=t['k'],
                t=args.t, 
                verbose=False
            )
            
            # Use ARI for scoring if labels exist
            if args.ground_truth and args.ground_truth in adata.obs:
                y_true = adata.obs[args.ground_truth].values
                score = 1 - metrics.adjusted_rand_score(y_true, y_pred)
            else:
                # Fallback to a composite of reconstruction and clustering loss
                score = network.model.history.history['loss'][-1]
            
            return {'loss': float(score), 'status': STATUS_OK}

        except Exception as e:
            # This ensures that even if one trial crashes, the whole hyperopt doesn't stop
            print(f"\n[!] Trial failed with error: {e}") 
            return {'loss': 1e10, 'status': STATUS_OK}

    # 6. Run Optimization
    print(f"Starting Hyperparameter Optimization for {args.hypern} trials...")
    trials = Trials()

    try:
        best = fmin(
            fn=objective, 
            space=hyper_params, 
            algo=tpe.suggest, 
            max_evals=args.hypern, 
            trials=trials
        )
    except KeyboardInterrupt:
        # If interrupted, we use the best trials found so far
        print("\nInterrupted! Saving current best results...")
        if len(trials.trials) > 0:
            # Get the best trial from the trials object
            best = trials.argmin 
        else:
            print("No trials completed. Exiting.")
            return None

    # 7. Map indices back to actual values for final save
    best_readable = {}
    for k, v in best.items():
        # Model & Training mappings
        if k == 'm_hidden': best_readable[k] = str(hidden_choices[v])
        elif k == 'm_act': best_readable[k] = act_choices[v]
        elif k == 't_batch': best_readable[k] = [64, 128, 256, 512][v]
        elif k == 't_opt': best_readable[k] = opt_choices[v]
        elif k == 'p_opt': best_readable[k] = opt_choices[v]
        
        # Clustering mappings
        elif k == 'c_num': best_readable[k] = [5, 10, 15, 20, 30][v]
        elif k == 'c_weights': best_readable[k] = str(weight_choices[v])
        elif k == 'c_ramp_m': best_readable[k] = [True, False][v]
        elif k == 'c_ramp_v': best_readable[k] = str(ramp_choices[v])
        
        # Topology mappings
        elif k == 't_size': best_readable[k] = [45, 64, 128][v]
        elif k == 't_in_mode': best_readable[k] = topo_input_choices[v]
        elif k == 't_lat_mode': best_readable[k] = topo_latent_choices[v]
        elif k == 't_k': best_readable[k] = [10, 15, 20, 30][v]
        
        # Continuous values
        else: best_readable[k] = float(v)

    # 8. Save results
    with open(os.path.join(output_dir, 'trials.pickle'), 'wb') as f:
        pickle.dump(trials, f)

    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump(best_readable, f, sort_keys=True, indent=4)

    # Final summary print
    print("\n" + "="*40)
    print("  SC-TOPODEC OPTIMIZATION COMPLETE  ")
    print("="*40)
    print(f"Results Directory: {output_dir}")
    print("Best Parameters Found:")
    print(json.dumps(best_readable, indent=4))
    print("="*40)

    return best_readable