import csv, os, random
import time
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, ops, optimizers
from sklearn.cluster import KMeans
from sklearn import metrics

from . import io
from .utils import compute_target_distribution
from .network import network_options
from .loss import soft_kmeans_loss
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


def dec_train(adata, network, output_dir=None, save_weights=True, save_interval=5, 
              optimizer='adam', learning_rate=0.01, epochs=300, update_interval=10, 
              batch_size=256, tol=1e-3, loss_weights=(1, 1, 0, 0), 
              use_raw_as_output=True, verbose=True, ground_truth=None, pretrain_epochs=200, 
              pretrain_optimizer='adam', pretrain_learning_rate=0.01, **kwds):
   
    model = network.model
    ae_loss_fn = network.loss 
    active_weights = loss_weights[:2] # We use only three losses for DEC
    clustering_layer = model.get_layer(name='clustering')
    
    # 1. Pretrain
    print("...Pretraining Autoencoder...")
    start_pretrain = time.time()

    network.model = network.zinb_ae # Temporarily point network.model to the AE version
    ae_train(adata, 
             network, 
             epochs=pretrain_epochs,
             optimizer=pretrain_optimizer,
             learning_rate=pretrain_learning_rate,
             verbose=verbose)
    network.model = model
    end_pretrain = time.time() 
    print(f"Pretraining complete in {end_pretrain - start_pretrain:.2f} seconds.")

    # 2. k-mean for centroid initialization
    print("...Initializing cluster centers with k-means...")
    kmeans = KMeans(n_clusters=network.n_clusters, n_init=20)
    latent_feat = network.encoder.predict({'count': adata.X, 
                                           'size_factors': adata.obs.size_factors.values})
    y_pred = kmeans.fit_predict(latent_feat)
    y_pred_last = np.copy(y_pred)
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    # 3. Setup Optimizer
    opt_dec = optimizers.get(optimizer)
    opt_dec.learning_rate = learning_rate

    @tf.function
    def train_step(x_counts, x_sf, y_p, y_raw):
        with tf.GradientTape() as tape:
            z = network.encoder({'count': x_counts, 'size_factors': x_sf})
            q, zinb_out = model({'count': x_counts, 'size_factors': x_sf})
            mu = clustering_layer.weights[0] # Cluster centers
            q = tf.clip_by_value(q, 1e-10, 1.0)

            l_zinb = ae_loss_fn(y_raw, zinb_out)
            l_kl = keras.losses.KLDivergence()(y_p, q)
            l_sk = soft_kmeans_loss(z, mu)

            total_loss = (loss_weights[0] * l_zinb) + \
                         (loss_weights[1] * l_kl) + \
                         (loss_weights[2] * l_sk)

        grads = tape.gradient(total_loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        opt_dec.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, l_zinb, l_kl, l_sk

    # 4. Iterative Training Loop
    print("...Training for clustering...")
    start_total_train = time.time()

    num_samples = adata.n_obs
    loss_vals = [0, 0, 0, 0] 

    for epoch in range(epochs):
        if epoch % update_interval == 0:
            q, _ = model.predict({'count': adata.X, 'size_factors': adata.obs.size_factors.values}, verbose=0)
            p = compute_target_distribution(q)
            y_pred = q.argmax(1)

            # --- Evaluation Block ---
            if ground_truth is not None and ground_truth in adata.obs:
                y_true = adata.obs[ground_truth].values
                acc = np.round(cluster_acc(y_true, y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
                l_print = [np.round(l, 5) for l in loss_vals]
                print(f'Epoch {epoch}: ACC={acc}, NMI={nmi}, ARI={ari}, '
                      f'Total_L={l_print[0]}, L_zinb={l_print[1]}, L_kl={l_print[2]}, L_sk={l_print[3]}')
            # -------------------------

            # Convergence Check
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if epoch > 0 and delta_label < tol:
                print(f'Converged at epoch {epoch}: delta {delta_label:.4f} < {tol}')
                break

        # Manual Batching
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            x_c = tf.cast(adata.X[batch_idx], tf.float32)
            x_s = adata.obs.size_factors.values[batch_idx]
            y_p_batch = tf.cast(p[batch_idx], tf.float32)
            y_r = adata.raw.X[batch_idx] if use_raw_as_output else adata.X[batch_idx]

            loss_vals = train_step(x_c, x_s, y_p_batch, y_r)

        if verbose:
            print(f"Epoch {epoch} - Total L: {loss_vals[0]:.4f}, L_kl: {loss_vals[1]:.4f}, "
                  f"L_zinb: {loss_vals[2]:.4f}, L_sk: {loss_vals[3]:.4f}")

    end_total_train = time.time()
    print(f"Total Clustering Training complete in {end_total_train - start_total_train:.2f} seconds.")

    return y_pred

def ramp_dec_train(adata, network, output_dir=None, save_weights=True, save_interval=5, 
                   optimizer='adam', learning_rate=0.001, epochs=300, update_interval=10, 
                   batch_size=256, tol=1e-3, loss_weights=(1, 1, 0.1, 0), use_raw_as_output=True,  
                   verbose=True, ground_truth=None, pretrain_epochs=200, pretrain_optimizer='adam',
                   pretrain_learning_rate=0.01, res_ramp=(0.1, 0.5, 1.0), **kwds):
   
    model = network.model
    ae_loss_fn = network.loss 
    clustering_layer = model.get_layer(name='clustering')
    
    # 1. Pretrain 
    print("...Pretraining Autoencoder...")
    start_pretrain = time.time()
    network.model = network.zinb_ae 
    ae_train(adata, network, epochs=pretrain_epochs, optimizer=pretrain_optimizer,
             learning_rate=pretrain_learning_rate, verbose=verbose)
    network.model = model
    print(f"Pretraining complete in {time.time() - start_pretrain:.2f}s")

    # 2. k-mean for centroid initialization
    print("...Initializing cluster centers with k-means...")
    kmeans = KMeans(n_clusters=network.n_clusters, n_init=20)
    latent_feat = network.encoder.predict({'count': adata.X, 
                                           'size_factors': adata.obs.size_factors.values})
    y_pred = kmeans.fit_predict(latent_feat)
    y_pred_last = np.copy(y_pred)
    clustering_layer.set_weights([kmeans.cluster_centers_])

    # 3. Setup Optimizer and Manual Train Step
    opt_dec = optimizers.get(optimizer)
    
    @tf.function
    def train_step(x_counts, x_sf, y_p, y_raw, current_weights):
        with tf.GradientTape() as tape:
            z = network.encoder({'count': x_counts, 'size_factors': x_sf})
            q, zinb_out = model({'count': x_counts, 'size_factors': x_sf})
            mu = clustering_layer.weights[0]

            l_zinb = ae_loss_fn(y_raw, zinb_out)
            l_kl = keras.losses.KLDivergence()(y_p, q)
            l_sk = soft_kmeans_loss(z, mu)

            # Apply the dynamic weights from the ramping phase
            total_loss = (current_weights[0] * l_zinb) + \
                         (current_weights[1] * l_kl) + \
                         (current_weights[2] * l_sk)

        grads = tape.gradient(total_loss, model.trainable_variables)
        opt_dec.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, l_zinb, l_kl, l_sk

    # 4. Iterative Ramping Loop
    print("...Training for clustering with Resolution Ramping...")
    start_total_train = time.time()
    num_samples = adata.n_obs
    loss_vals = [0, 0, 0, 0]

    for res_idx, current_res in enumerate(res_ramp):
        print(f"\n>>> Phase {res_idx+1}: Scaling Clustering/SoftK weights by {current_res}")
        
        # Scale KL and SoftK weights by the current resolution factor
        current_weights = [
            loss_weights[0],             # ZINB stays constant
            loss_weights[1] * current_res, # KL scales
            loss_weights[2] * current_res  # SoftK scales
        ]
        
        # Anneal learning rate for each phase
        opt_dec.learning_rate = learning_rate / (res_idx + 1)
        
        epochs_per_phase = epochs // len(res_ramp)

        for epoch in range(epochs_per_phase):
            if epoch % update_interval == 0:
                q, _ = model.predict({'count': adata.X, 'size_factors': adata.obs.size_factors.values}, verbose=0)
                p = compute_target_distribution(q)
                y_pred = q.argmax(1)

                if ground_truth is not None and ground_truth in adata.obs:
                    y_true = adata.obs[ground_truth].values
                    acc = np.round(cluster_acc(y_true, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
                    print(f'Res {current_res} | Ep {epoch}: ACC={acc}, NMI={nmi}')

                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if epoch > 0 and delta_label < tol:
                    print(f'Converged at resolution {current_res}')
                    break

            # Batch Training
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                x_c = adata.X[batch_idx]
                x_s = adata.obs.size_factors.values[batch_idx]
                y_p_batch = p[batch_idx]
                y_r = adata.raw.X[batch_idx] if use_raw_as_output else adata.X[batch_idx]

                loss_vals = train_step(x_c, x_s, y_p_batch, y_r, current_weights)

            if verbose:
                print(f"Res {current_res} Ep {epoch} - Total L: {loss_vals[0]:.4f}, L_sk: {loss_vals[3]:.4f}")

    print(f"Ramp Training complete in {time.time() - start_total_train:.2f}s")
    return y_pred
