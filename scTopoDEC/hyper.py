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
    Bayesian Hyperparameter Optimization for scRNA-seq Autoencoders and scDEC.
    Designed for Keras 3 and Hyperopt.
    """
    # 1. Reproducibility setup
    keras.utils.set_random_seed(42)
    
    output_dir = os.path.join(args.outputdir, 'hyperopt_results')
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Dataset
    adata = io.read_dataset(args.input, 
                            transpose=args.transpose, 
                            test_split=False)

    # 3. Define Search Space Choices
    hidden_choices = [
        (256, 64, 32, 64, 256), 
        (128, 64, 32, 64, 128), 
        (64, 32, 64), 
        (128, 64, 128)
    ]
    act_choices = ['relu', 'selu', 'elu', 'PReLU', 'LeakyReLU']
    ae_choices = ['ae', 'zinb', 'dec']
    weight_choices = [[1, 1], [0.1, 1], [1, 0.1], [1, 0.5]]

    hyper_params = {
        "data": {
            "norm_input_log": hp.choice('d_norm_log', (True, False)),
            "norm_input_zeromean": hp.choice('d_norm_zeromean', (True, False)),
            "norm_input_sf": hp.choice('d_norm_sf', (True, False)),
        },
        "model": {
            "lr": hp.loguniform("m_lr", np.log(1e-4), np.log(1e-2)),
            "ridge": hp.loguniform("m_ridge", np.log(1e-7), np.log(1e-1)),
            "l1_enc_coef": hp.loguniform("m_l1_enc_coef", np.log(1e-7), np.log(1e-1)),
            "hidden_size": hp.choice("m_hiddensize", hidden_choices),
            "activation": hp.choice("m_activation", act_choices),
            "aetype": hp.choice("m_aetype", ae_choices),
            "batchnorm": hp.choice("m_batchnorm", (True, False)),
            "dropout": hp.uniform("m_do", 0, 0.5),
            "input_dropout": hp.uniform("m_input_do", 0, 0.5),
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

    # 4. Data Preparation Helper
    def data_fn(norm_input_log, norm_input_zeromean, norm_input_sf):
        ad = adata.copy()
        ad = io.normalize(ad,
                          size_factors=norm_input_sf,
                          logtrans_input=norm_input_log,
                          normalize_input=norm_input_zeromean)
        
        x_train = {
            'count': ad.X, 
            'size_factors': ad.obs.size_factors.values
        }
        y_train = ad.raw.X if ad.raw else ad.X
        return (x_train, y_train)

    # 5. Objective Function (The Trial Runner)
    def objective(params):
        keras.backend.clear_session()
        
        d_p, m_p, f_p = params['data'], params['model'], params['fit']
        c_p = params.get('clustering')

        try:
            # Prepare data for this trial
            (x_train, y_train) = data_fn(d_p['norm_input_log'], 
                                         d_p['norm_input_zeromean'], 
                                         d_p['norm_input_sf'])

            # Define common arguments used by ALL models
            model_kwargs = {
                "input_size": y_train.shape[1],
                "hidden_size": m_p['hidden_size'],
                "l1_enc_coef": m_p['l1_enc_coef'],
                "ridge": m_p['ridge'],
                "hidden_dropout": m_p['dropout'],
                "input_dropout": m_p['input_dropout'],
                "batchnorm": m_p['batchnorm'],
                "activation": m_p['activation']
            }

            # Add clustering-specific arguments if the type is 'dec'
            if m_p['aetype'] == 'dec':
                model_kwargs["n_clusters"] = c_p['n_clusters']
                model_kwargs["alpha"] = c_p['alpha']

            network = network_options[m_p['aetype']](**model_kwargs)
            network.build()

            # Execute Training
            if m_p['aetype'] == 'dec':
                # Iterative Clustering Mode
                y_pred = dec_train(
                    adata, 
                    network, 
                    epochs=f_p['epochs'], 
                    loss_weights=c_p['loss_weights'],
                    optimizer=optimizers.Adam(learning_rate=m_p['lr'], clipvalue=5.0),
                    verbose=False,
                    save_weights=False
                )
                
                # Scoring: Prioritize ARI if ground truth is available
                if args.ground_truth and args.ground_truth in adata.obs:
                    y_true = adata.obs[args.ground_truth].values
                    score = 1 - metrics.adjusted_rand_score(y_true, y_pred)
                else:
                    score = network.model.history.history['loss'][-1]
            
            else:
                # Standard AE/ZINB Mode
                opt = optimizers.Adam(learning_rate=m_p['lr'], clipvalue=5.0)
                network.model.compile(loss=network.loss, optimizer=opt)
                history = network.model.fit(
                    x_train, y_train,
                    epochs=f_p['epochs'],
                    batch_size=f_p['batch_size'],
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[keras.callbacks.TerminateOnNaN()]
                )
                score = np.min(history.history['val_loss'])

            if np.isnan(score):
                return {'loss': 1e10, 'status': STATUS_OK}
                
            return {'loss': float(score), 'status': STATUS_OK}

        except Exception as e:
            print(f"Trial failed with error: {e}")
            return {'loss': 1e10, 'status': STATUS_OK}

    # 6. Run Optimization
    print(f"Starting Hyperparameter Optimization for {args.hypern} trials...")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=hyper_params,
        algo=tpe.suggest,
        max_evals=args.hypern,
        trials=trials,
        catch_eval_exceptions=True
    )

    # 7. Map indices back to actual values for final save
    best_readable = {}
    for k, v in best.items():
        if k == 'm_hiddensize': best_readable[k] = str(hidden_choices[v])
        elif k == 'm_activation': best_readable[k] = act_choices[v]
        elif k == 'm_aetype': best_readable[k] = ae_choices[v]
        elif k == 'c_n_clusters': best_readable[k] = [5, 10, 15, 20, 30][v]
        elif k == 'c_loss_weights': best_readable[k] = weight_choices[v]
        elif k.startswith('d_norm'): best_readable[k] = bool(v)
        else: best_readable[k] = float(v)

    # 8. Save results
    with open(os.path.join(output_dir, 'trials.pickle'), 'wb') as f:
        pickle.dump(trials, f)

    with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
        json.dump(best_readable, f, sort_keys=True, indent=4)

    print("\n" + "="*30)
    print("Optimization Finished Successfully")
    print(f"Best Configuration saved to: {output_dir}/best_config.json")
    print(json.dumps(best_readable, indent=4))
    print("="*30)