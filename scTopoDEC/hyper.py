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
from .train import dec_train, ae_train 

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
        (256, 128, 64, 128, 256), 
        (256, 128, 32, 128, 256), 
        (256, 64, 32, 64, 256), 
        (128, 64, 32, 64, 128),
        (256, 64, 256),
        (256, 32, 256),
        (128, 64, 128),
        (128, 32, 128),
        (64, 32, 64)
    ]
    act_choices = ['relu', 'selu', 'elu', 'PReLU', 'LeakyReLU']

    # 4-weight tuple: (ZINB, KL, SoftK, Topo)
    weight_choices = [
        (1.0, 1.0, 0.0, 0.1), 
        (1.0, 1.0, 0.0, 1.0),
        (0.5, 1.0, 0.0, 0.5),
        (1.0, 1.0, 0.1, 1.0)
    ]

    hyper_params = {
        "data": {
            "norm_input_log": hp.choice('d_norm_log', (True, False)),
            "norm_input_zeromean": hp.choice('d_norm_zeromean', (True, False)),
            "norm_input_sf": hp.choice('d_norm_sf', (True, False)),
        },
        "model": {
            "lr": hp.loguniform("m_lr", np.log(1e-4), np.log(1e-2)),
            "hidden_size": hp.choice("m_hiddensize", hidden_choices),
            "activation": hp.choice("m_activation", act_choices),
            "batchnorm": hp.choice("m_batchnorm", (True, False)),
            "dropout": hp.uniform("m_do", 0, 0.5),
        },
        "topology": {
            "homology_dim": hp.choice("t_dim", (0, 1)),
            "max_edge": hp.uniform("t_max_edge", 0.5, 5.0),
        },
        "clustering": {
            "n_clusters": hp.choice("c_n_clusters", (5, 10, 15, 20, 30)),
            "alpha": hp.uniform("c_alpha", 0.5, 2.0),
            "loss_weights": hp.choice("c_loss_weights", weight_choices)
        },
        "fit": {
            "epochs": args.hyperepoch,
            "batch_size": 256
        }
    }

    # 4. Objective Function (The Trial Runner)
    def objective(params):
        keras.backend.clear_session()
        # GUDHI requires eager mode; ensure graph mode isn't forced by a hidden @tf.function
        tf.config.run_functions_eagerly(True) 
        
        d_p, m_p, f_p = params['data'], params['model'], params['fit']
        c_p, t_p = params['clustering'], params['topology']

        try:
            # Prepare data
            ad = adata.copy()
            ad = io.normalize(ad, size_factors=d_p['norm_input_sf'], 
                             logtrans_input=d_p['norm_input_log'], 
                             normalize_input=d_p['norm_input_zeromean'])

            model_kwargs = {
                "input_size": ad.n_vars,
                "hidden_size": m_p['hidden_size'],
                "hidden_dropout": m_p['dropout'],
                "batchnorm": m_p['batchnorm'],
                "activation": m_p['activation'],
                "n_clusters": c_p['n_clusters'],
                "alpha": c_p['alpha']
            }

            network = network_options['dec'](**model_kwargs)
            network.build()

            # Execute training with topology
            y_pred = dec_train(
                ad, 
                network, 
                epochs=f_p['epochs'], 
                loss_weights=c_p['loss_weights'],
                optimizer=optimizers.Adam(learning_rate=m_p['lr'], clipvalue=5.0),
                homology_dim=t_p['homology_dim'],
                maximum_edge_length=t_p['max_edge'],
                verbose=False,
                save_weights=False
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
            print(f"Trial failed: {e}")
            return {'loss': 1e10, 'status': STATUS_OK}

    # 6. Run Optimization
    print(f"Starting Hyperparameter Optimization for {args.hypern} trials...")
    trials = Trials()
    best = fmin(fn=objective, space=hyper_params, algo=tpe.suggest, max_evals=args.hypern, trials=trials)

    # 7. Map indices back to actual values for final save
    best_readable = {}
    for k, v in best.items():
        if k == 'm_hiddensize': best_readable[k] = str(hidden_choices[v])
        elif k == 'm_activation': best_readable[k] = act_choices[v]
        elif k == 'c_n_clusters': best_readable[k] = [5, 10, 15, 20, 30][v]
        elif k == 'c_loss_weights': best_readable[k] = str(weight_choices[v])
        elif k == 't_dim': best_readable[k] = [0, 1][v]
        elif k.startswith('d_norm'): best_readable[k] = bool(v)
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