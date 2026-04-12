import csv, os, random
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, ops, optimizers
from sklearn.cluster import KMeans
from sklearn import metrics

from . import io
from .utils import compute_target_distribution
from .network import network_options
from .metric import cluster_acc


def ae_train(adata, network, output_dir=None, optimizer='adam', learning_rate=0.001,
          initial_weights=None, epochs=200, reduce_lr=10, output_subset=None, 
          use_raw_as_output=True, early_stop=15, batch_size=256, clip_grad=1.0, save_weights=True,
          validation_split=0.1, tensorboard=False, verbose=True, **kwds):
   
    model = network.model
    loss_fn = network.loss
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    opt_class = optimizers.get(optimizer).__class__
    opt = opt_class(learning_rate=learning_rate, clipnorm=clip_grad)
    model.compile(loss=loss_fn, optimizer=opt)

    if initial_weights and os.path.exists(initial_weights):
        if verbose:
            print(f"Restoring weights from: {initial_weights}")
        network.load_weights(initial_weights) 
    elif initial_weights:
        print(f"Warning: {initial_weights} not found. Training from scratch.")

    callbacks = []

    if save_weights and output_dir is not None:
        ckpt_path = os.path.join(output_dir, "weights.weights.h5")
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            verbose=verbose,
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss'
        )
        callbacks.append(checkpointer)

    if reduce_lr:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=reduce_lr, verbose=verbose, factor=0.1))
            
    if early_stop:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=early_stop, verbose=verbose, restore_best_weights=True))

    if tensorboard and output_dir:
        callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(output_dir, 'tb')))

    if verbose:
        model.summary()

    inputs = {
        'count': adata.X, 
        'size_factors': adata.obs.size_factors.values
    }

    if output_subset:
        gene_idx = [adata.var_names.get_loc(x) for x in output_subset]
        target = adata.raw.X[:, gene_idx] if use_raw_as_output else adata.X[:, gene_idx]
    else:
        target = adata.raw.X if use_raw_as_output else adata.X

    history = model.fit(
        inputs, 
        target,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        validation_split=validation_split,
        verbose=verbose,
        **kwds
    )

    return history


def ae_train_with_args(args):
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    keras.utils.set_random_seed(42)  
    os.environ['PYTHONHASHSEED'] = '0'

    # If perform hyperparams -> exit
    if args.hyper:
        hyper(args)
        return

    from .hyper import hyper

    adata = io.read_dataset(args.input,
                            transpose=(not args.transpose),
                            check_counts=args.checkcounts,
                            test_split=args.testsplit)

    adata = io.normalize(adata,
                         size_factors=args.sizefactors,
                         logtrans_input=args.loginput,
                         normalize_input=args.norminput)

    if args.denoisesubset:
        genelist = list(set(io.read_genelist(args.denoisesubset)))

        if not all(g in adata.var_names for g in genelist):
             raise ValueError('Gene list is not overlapping with dataset')

        output_size = len(genelist)

    else:
        genelist = None
        output_size = adata.n_vars

    hidden_size = [int(x) for x in args.hiddensize.split(',')]
    hidden_dropout = [float(x) for x in args.dropoutrate.split(',')]

    if len(hidden_dropout) == 1:
        hidden_dropout = hidden_dropout[0]

    input_size = adata.n_vars
    
    net = network_options[args.type](
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        l2_coef=args.l2,
        l1_coef=args.l1,
        l2_enc_coef=args.l2enc,
        l1_enc_coef=args.l1enc,
        ridge=args.ridge,
        hidden_dropout=hidden_dropout,
        input_dropout=args.inputdropout,
        batchnorm=args.batchnorm,
        activation=args.activation,
        init=args.init,
        debug=args.debug,
        file_path=args.outputdir
    )

    net.save() # Saves metadata
    net.build()

    history = train(
        adata[adata.obs.dca_split == 'train'], 
        net,
        output_dir=args.outputdir,
        learning_rate=args.learningrate,
        epochs=args.epochs, 
        batch_size=args.batchsize,
        early_stop=args.earlystop,
        reduce_lr=args.reducelr,
        output_subset=genelist,
        optimizer=args.optimizer,
        clip_grad=args.gradclip,
        save_weights=args.saveweights,
        initial_weights=args.loadweights,
        tensorboard=args.tensorboard
    )

    if genelist:
        predict_columns = adata.var_names[adata.var_names.isin(genelist)]
    else:
        predict_columns = adata.var_names

    net.predict(adata, mode='full', return_info=True)


def dec_train(adata, network, output_dir=None, save_weights=True, save_interval=5, 
              optimizer='adam', learning_rate=0.01, epochs=300, update_interval=10, 
              batch_size=256, tol=1e-3, loss_weights=[1, 1, 0], use_raw_as_output=True,  
              verbose=True, ground_truth=None, pretrain_epochs=200, pretrain_optimizer='adam',
              pretrain_learning_rate=0.01, **kwds):
   
    model = network.model
    ae_loss = network.loss # ZINB loss
    active_weights = loss_weights[:2] # We use only two losses for DEC

    # 1. Pretrain
    print("...Pretraining Autoencoder...")
    network.model = network.zinb_ae # Temporarily point network.model to the AE version
    ae_train(adata, 
             network, 
             epochs=pretrain_epochs,
             optimizer=pretrain_optimizer,
             learning_rate=pretrain_learning_rate,
             verbose=verbose)
    network.model = model

    # 2. k-mean for centroid initialization
    print("...Initializing cluster centers with k-means...")
    kmeans = KMeans(n_clusters=network.n_clusters, n_init=20)
    latent_feat = network.encoder.predict({'count': adata.X, 
                                           'size_factors': adata.obs.size_factors.values})
    y_pred = kmeans.fit_predict(latent_feat)
    y_pred_last = np.copy(y_pred)
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    # 3. Compile Model with Multiple Outputs
    opt_dec = optimizers.get(optimizer)
    opt_dec.learning_rate = learning_rate
    opt_dec.clipnorm = 1.0  
    model.compile(loss=['kld', ae_loss], loss_weights=active_weights, optimizer=opt_dec)

    # 4. Iterative DEC training
    print("...Training for clustering...")
    losses = [0.0, 0.0, 0.0]
    for epoch in range(epochs):
        # Update target distribution 'p' every 'update_interval' epochs
        if epoch % update_interval == 0:
            q, _ = model.predict({'count': adata.X, 
                                  'size_factors': adata.obs.size_factors.values}, verbose=0)
            p = compute_target_distribution(q) 
            y_pred = q.argmax(1) # Predicted cluster

            # --- Evaluation Block ---
            if ground_truth is not None and ground_truth in adata.obs:
                y_true = adata.obs[ground_truth].values
                acc = np.round(cluster_acc(y_true, y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
                l_print = [np.round(l, 5) for l in losses]
                print(f'Epoch {epoch}: ACC={acc}, NMI={nmi}, ARI={ari}, '
                      f'Total_L={l_print[0]}, Lc={l_print[1]}, Lr={l_print[2]}')
            # -------------------------

            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            
            if epoch > 0 and delta_label < tol:
                print(f'Converged at epoch {epoch}: delta {delta_label:.4f} < {tol}')
                break

        # Train for one epoch with current 'p'
        # Target Y is a list: [p, raw_counts] matching the model outputs
        history = model.fit(
            x={'count': adata.X, 'size_factors': adata.obs.size_factors.values},
            y=[p, adata.raw.X if use_raw_as_output else adata.X],
            epochs=1,
            batch_size=batch_size,
            validation_split=0.0, 
            shuffle=True,
            verbose=verbose,
            **kwds
        )

        # Note: 'total_loss' is Total_L, 'clustering_loss' is Lc, 'slice_loss' (or similar) is Lr
        h = history.history
        keys = list(h.keys()) 
        l_total = h.get('loss', [0])[-1]
        l_clust = h.get('clustering_loss', h.get(keys[1], [0]))[-1]
        l_recon = h.get('zinb_loss', h.get(keys[2], [0]))[-1]
        losses = [l_total, l_clust, l_recon]

        if save_weights and output_dir and epoch % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f'dec_weights_epoch_{epoch}.weights.h5')
            model.save_weights(ckpt_path)
            if verbose: print(f"Checkpoint saved: {ckpt_path}")

    # 5. Save final weights
    if save_weights and output_dir:
        final_path = os.path.join(output_dir, 'dec_model_final.weights.h5')
        model.save_weights(final_path)
        print(f"Final model saved to: {final_path}")

    return y_pred

def ramp_dec_train(adata, network, output_dir=None, save_weights=True, save_interval=5, 
                   optimizer='adam', learning_rate=0.001, epochs=300, update_interval=10, 
                   batch_size=256, tol=1e-3, loss_weights=[1, 1, 0], use_raw_as_output=True,  
                   verbose=True, ground_truth=None, pretrain_epochs=200, pretrain_optimizer='adam',
                   pretrain_learning_rate=0.01, res_ramp=[0.1, 0.5, 1.0], **kwds):
   
    model = network.model
    ae_loss = network.loss 
    
    # 1. Pretrain 
    print("...Pretraining Autoencoder...")
    network.model = network.zinb_ae 
    ae_train(adata, network, epochs=pretrain_epochs, optimizer=pretrain_optimizer,
             learning_rate=pretrain_learning_rate, verbose=verbose)
    network.model = model

    # 2. k-mean for centroid initialization
    print("...Initializing cluster centers with k-means...")
    kmeans = KMeans(n_clusters=network.n_clusters, n_init=20)
    latent_feat = network.encoder.predict({'count': adata.X, 
                                           'size_factors': adata.obs.size_factors.values})
    y_pred = kmeans.fit_predict(latent_feat)
    y_pred_last = np.copy(y_pred)
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    # 3. Iterative ramping loop in DEC
    print("...Training for clustering...")
    for res_idx, current_res in enumerate(res_ramp):
        print(f"\n>>> Iterative Ramping Phase {res_idx+1}: Resolution Scaling at {current_res}")
        
        # Adjust active weights: Gradually increase importance of Clustering (Lc)
        active_weights = [loss_weights[0], loss_weights[1] * current_res]

        opt_dec = optimizers.get(optimizer)
        opt_dec.learning_rate = learning_rate / (res_idx + 1) # Annealing learning rate
        model.compile(loss=['kld', ae_loss], loss_weights=active_weights, optimizer=opt_dec)

        losses = [0.0, 0.0, 0.0]
        
        # Standard DEC Loop within the current resolution stage
        for epoch in range(epochs // len(res_ramp)):
            if epoch % update_interval == 0:
                q, _ = model.predict({'count': adata.X, 
                                      'size_factors': adata.obs.size_factors.values}, verbose=0)
                
                # DESC Benefit: Sharper Target Distribution
                p = compute_target_distribution(q) 
                y_pred = q.argmax(1)

                # --- Evaluation Block ---
                if ground_truth is not None and ground_truth in adata.obs:
                    y_true = adata.obs[ground_truth].values
                    acc = np.round(cluster_acc(y_true, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
                    l_print = [np.round(l, 5) for l in losses]
                    print(f'Res {current_res} | Epoch {epoch}: ACC={acc}, NMI={nmi}, ARI={ari}, '
                          f'Total_L={l_print[0]}, Lc={l_print[1]}, Lr={l_print[2]}')
                # -------------------------

                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                
                if epoch > 0 and delta_label < tol:
                    print(f'Resolution {current_res}: delta {delta_label:.4f} < {tol}')
                    break

            history = model.fit(
                x={'count': adata.X, 'size_factors': adata.obs.size_factors.values},
                y=[p, adata.raw.X if use_raw_as_output else adata.X],
                epochs=1,
                batch_size=batch_size,
                shuffle=True,
                verbose=verbose,
                **kwds
            )
            
            # Note: 'total_loss' is Total_L, 'clustering_loss' is Lc, 'slice_loss' (or similar) is Lr
            h = history.history
            keys = list(h.keys()) 
            l_total = h.get('loss', [0])[-1]
            l_clust = h.get('clustering_loss', h.get(keys[1], [0]))[-1]
            l_recon = h.get('zinb_loss', h.get(keys[2], [0]))[-1]
            losses = [l_total, l_clust, l_recon]

            if save_weights and output_dir and epoch % save_interval == 0:
                ckpt_path = os.path.join(output_dir, f'dec_weights_epoch_{epoch}.weights.h5')
                model.save_weights(ckpt_path)
                if verbose: print(f"Checkpoint saved: {ckpt_path}")

    # 5. Final Save
    if save_weights and output_dir:
        model.save_weights(os.path.join(output_dir, 'dec_model_final.weights.h5'))

    return y_pred
