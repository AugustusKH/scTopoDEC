import csv, os, random
import time
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, ops, optimizers
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import metrics

from . import io
from .utils import compute_target_distribution, get_topo_representation
from .network import network_options
from .layers import RipsLayer
from .loss import soft_kmeans_loss, topo_loss
from .metric import cluster_acc


# ==============================================================================
# Global train step
# ==============================================================================

#@tf.function
def train_step(x_counts, x_sf, y_p, y_raw, network, model, clustering_layer, 
               ae_loss_fn, opt_dec, loss_weights, topo_size, pg_dist, order,
               topo_latent_mode, k, t, topo_input_batch=None, rips_layer=None):
    """
    Executes a single training iteration (batch update) for scTopoDEC.
    
    This function calculates a joint loss across reconstruction, clustering, 
    and topological manifold preservation.
    """
    with tf.GradientTape() as tape:
        z = network.encoder({'count': x_counts, 'size_factors': x_sf})
        q, zinb_out = model({'count': x_counts, 'size_factors': x_sf})
        mu = clustering_layer.weights[0] 
        
        # Stability clipping
        q = tf.clip_by_value(q, 1e-10, 1.0)
        y_p = tf.clip_by_value(y_p, 1e-7, 1.0)

        # Base losses
        l_zinb = ae_loss_fn(y_raw, zinb_out)
        l_kl = keras.losses.KLDivergence()(y_p, q)

        # Optional soft k-mean loss
        l_sk = tf.constant(0.0, dtype=tf.float32)
        if loss_weights[2] > 0:
            l_sk = soft_kmeans_loss(z, mu)

        # Optional topological loss
        l_topo = tf.constant(0.0, dtype=tf.float32)
        if loss_weights[3] > 0 and topo_input_batch is not None:
            z_topo = get_topo_representation(z, latent_mode=topo_latent_mode, 
                                             k=k, t=t, is_latent=True)
            l_topo = topo_loss(topo_input_batch, z_topo, rips_layer, topo_size,
                               pg_dist=pg_dist, order=order)

        # Total loss calculation 
        total_loss = (loss_weights[0] * l_zinb) + \
                     (loss_weights[1] * l_kl) + \
                     (loss_weights[2] * l_sk) + \
                     (loss_weights[3] * l_topo)

    # Gradient Descent
    tf.debugging.assert_all_finite(total_loss, "Loss became NaN")
    grads = tape.gradient(total_loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    opt_dec.apply_gradients(zip(grads, model.trainable_variables))
    
    return total_loss, l_zinb, l_kl, l_sk, l_topo


# ==============================================================================
# Pretraining (ZINB-autoencoder)
# ==============================================================================

def pretrain(adata, network, output_dir=None, optimizer='adam', learning_rate=0.001,
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


# ==============================================================================
# Clustering (DEC)
# ==============================================================================

def train(adata, network, output_dir=None, save_weights=True, save_interval=5, 
          optimizer='adam', learning_rate=0.01, epochs=300, update_interval=10, 
          batch_size=256, tol=1e-3, loss_weights=(1, 1, 0, 0), use_raw_as_output=True, 
          verbose=True, ground_truth=None, pretrain_epochs=200, pretrain_optimizer='adam', 
          pretrain_learning_rate=0.01, reduce_lr_patience=10, early_stop_patience=15, 
          cluster_early_stop=False, homology_dim=1, maximum_edge_length=2., 
          topo_size=64, pg_dist='wd', order=1., topo_input_mode='pca', 
          topo_latent_mode='raw', n_components=30, k=15, t=8, **kwds):
   
    model = network.model
    ae_loss_fn = network.loss 
    clustering_layer = model.get_layer(name='clustering')
    topo_input = None
    rips_layer = None
    
    # 1. Pretrain
    print("\n...Pretraining Autoencoder...")
    start_pretrain = time.time()

    network.model = network.zinb_ae # Temporarily point network.model to the AE version
    pretrain(adata, 
             network, 
             epochs=pretrain_epochs,
             optimizer=pretrain_optimizer,
             learning_rate=pretrain_learning_rate,
             verbose=verbose)
    network.model = model
    end_pretrain = time.time() 
    print(f"Pretraining complete in {end_pretrain - start_pretrain:.2f} seconds.")

    # 2. k-mean for centroid initialization
    print("\n...Initializing cluster centers with k-means...")
    kmeans = KMeans(n_clusters=network.n_clusters, n_init=20)
    latent_feat = network.encoder.predict({'count': adata.X, 
                                           'size_factors': adata.obs.size_factors.values})
    y_pred = kmeans.fit_predict(latent_feat)
    y_pred_last = np.copy(y_pred)
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    # 3. Setup Optimizer
    opt_dec = optimizers.get(optimizer)
    opt_dec.build(model.trainable_variables)
    opt_dec.learning_rate = learning_rate

    # 4. Iterative Training Loop
    print("\n...Training for clustering...")
    start_total_train = time.time()

    num_samples = adata.n_obs
    loss_vals = [0, 0, 0, 0, 0] 
    
    # Define ReduceLROnPlateau and EarlyStopping variables
    best_loss = np.inf
    best_weights = None
    wait, es_wait = 0, 0
    factor = 0.1

    # Contruct the RipsLayer and generate topological input representation
    if loss_weights[3] > 0:
        topo_input = get_topo_representation(adata, input_mode=topo_input_mode, n_components=n_components, 
                                             k=k, t=t, is_latent=False)
        rips_layer = RipsLayer(
            maximum_edge_length=maximum_edge_length, 
            homology_dimensions=[homology_dim]
            )

    for epoch in tqdm(range(epochs), desc="Training DEC", unit="epoch"):
        if epoch % update_interval == 0:
            q, _ = model.predict({'count': adata.X, 'size_factors': adata.obs.size_factors.values}, verbose=0)
            p = compute_target_distribution(q)
            y_pred = q.argmax(1)

            # --- Evaluation Block ---
            if verbose and ground_truth is not None and ground_truth in adata.obs:
                y_true = adata.obs[ground_truth].values
                acc = np.round(cluster_acc(y_true, y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
                l_print = [np.round(l, 5) for l in loss_vals]
                print(f"Epoch {epoch}: ACC={acc}, NMI={nmi}, ARI={ari}, Total_L={l_print[0]}")
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
            x_s = tf.cast(adata.obs.size_factors.values[batch_idx], tf.float32)
            topo_input_batch = None
            y_p_batch = tf.cast(p[batch_idx], tf.float32)
            y_r = adata.raw.X[batch_idx] if use_raw_as_output else adata.X[batch_idx]

            if topo_input is not None:
                topo_input_batch = tf.cast(topo_input[batch_idx], tf.float32)

            loss_vals = train_step(
                x_c, x_s, y_p_batch, y_r, 
                network=network, 
                model=model, 
                clustering_layer=clustering_layer,
                ae_loss_fn=ae_loss_fn, 
                opt_dec=opt_dec, 
                loss_weights=loss_weights,
                topo_size=topo_size,
                pg_dist=pg_dist, order=order,
                topo_latent_mode=topo_latent_mode,
                k=k, t=t, topo_input_batch=topo_input_batch,
                rips_layer=rips_layer
            )

        if verbose:
            print(f"Epoch {epoch} - Total L: {loss_vals[0]:.4f}, L_zinb: {loss_vals[1]:.4f}, "
                  f"L_kl: {loss_vals[2]:.4f}, L_sk: {loss_vals[3]:.4f}, L_topo: {loss_vals[4]:.4f}")
            
        current_loss = loss_vals[0]

        # --- Logic for Best Weights, ReduceLR, and EarlyStopping ---
        if current_loss < best_loss - tol:
            best_loss = current_loss
            best_weights = [tf.identity(w) for w in model.get_weights()]
            wait, es_wait = 0, 0
        else:
            wait += 1
            es_wait += 1
            
            if wait >= reduce_lr_patience:
                old_lr = float(opt_dec.learning_rate)
                new_lr = old_lr * factor
                opt_dec.learning_rate = new_lr

                if verbose:
                    print(f"\nEpoch {epoch}: ReduceLROnPlateau reducing learning rate to {new_lr:.3e}")
                wait = 0

            if es_wait >= early_stop_patience and cluster_early_stop:
                print(f"Epoch {epoch}: Early stopping triggered.")
                break

    # --- RESTORE BEST WEIGHTS ---
    if best_weights is not None:
        if verbose: print("Restoring best weights from training...")
        model.set_weights(best_weights)

    end_total_train = time.time()
    print(f"Total Clustering Training complete in {end_total_train - start_total_train:.2f} seconds.")

    return y_pred


def ramp_train(adata, network, output_dir=None, save_weights=True, save_interval=5, 
               optimizer='adam', learning_rate=0.001, epochs=300, update_interval=10, 
               batch_size=256, tol=1e-3, loss_weights=(1, 1, 0.1, 0), use_raw_as_output=True,  
               verbose=True, ground_truth=None, pretrain_epochs=200, pretrain_optimizer='adam',
               pretrain_learning_rate=0.01, res_ramp=(0.1, 0.5, 1.0), early_stop_patience=15, 
               cluster_early_stop=False, homology_dim=1, maximum_edge_length=2., topo_size=64, 
               pg_dist='wd', order=1., topo_input_mode='pca', topo_latent_mode='raw', n_components=30, 
               k=15, t=8, **kwds):
   
    model = network.model
    ae_loss_fn = network.loss 
    clustering_layer = model.get_layer(name='clustering')
    topo_input = None
    rips_layer = None
    
    # 1. Pretrain 
    print("\n...Pretraining Autoencoder...")
    start_pretrain = time.time()
    network.model = network.zinb_ae 
    pretrain(adata, network, epochs=pretrain_epochs, optimizer=pretrain_optimizer,
             learning_rate=pretrain_learning_rate, verbose=verbose)
    network.model = model
    print(f"Pretraining complete in {time.time() - start_pretrain:.2f}s")

    # 2. k-mean for centroid initialization
    print("\n...Initializing cluster centers with k-means...")
    kmeans = KMeans(n_clusters=network.n_clusters, n_init=20)
    latent_feat = network.encoder.predict({'count': adata.X, 
                                           'size_factors': adata.obs.size_factors.values})
    y_pred = kmeans.fit_predict(latent_feat)
    y_pred_last = np.copy(y_pred)
    clustering_layer.set_weights([kmeans.cluster_centers_])

    # 3. Setup Optimizer and Manual Train Step
    opt_dec = optimizers.get(optimizer)
    opt_dec.learning_rate = learning_rate
    opt_dec.build(model.trainable_variables)

    # Contruct the RipsLayer
    if loss_weights[3] > 0:
        topo_input = get_topo_representation(adata, input_mode=topo_input_mode, n_components=n_components, 
                                             k=k, t=t, is_latent=False)
        rips_layer = RipsLayer(
            maximum_edge_length=maximum_edge_length, 
            homology_dimensions=[homology_dim]
            )

    # 4. Iterative Ramping Loop
    print("\n...Training for clustering with Resolution Ramping...")
    start_total_train = time.time()
    num_samples = adata.n_obs
    loss_vals = [0, 0, 0, 0, 0]

    # Define ReduceLROnPlateau and EarlyStopping variables
    best_loss = np.inf
    best_weights = None

    for res_idx, current_res in enumerate(res_ramp):
        print(f"\n>>> Phase {res_idx+1}: Scaling Clustering/SoftK weights by {current_res}")

        # Reset patience and best loss for each new resolution phase
        es_wait = 0
        phase_best_loss = np.inf
        
        # Scale KL and SoftK weights by the current resolution factor
        current_weights = [
            loss_weights[0],                # ZINB stays constant
            loss_weights[1] * current_res,  # KL scales
            loss_weights[2] * current_res,  # SoftK scales
            loss_weights[3]                 # Topo
        ]
        
        # Anneal learning rate for each phase
        opt_dec.learning_rate = learning_rate / (res_idx + 1)
        epochs_per_phase = epochs // len(res_ramp)

        pbar = tqdm(range(epochs_per_phase), desc=f"Res {current_res}", unit="epoch")
        for epoch in pbar: 
            if epoch % update_interval == 0:
                q, _ = model.predict({'count': adata.X, 'size_factors': adata.obs.size_factors.values}, verbose=0)
                p = compute_target_distribution(q)
                y_pred = q.argmax(1)

                # --- Evaluation block ---
                if verbose and ground_truth is not None and ground_truth in adata.obs:
                    y_true = adata.obs[ground_truth].values
                    acc = np.round(cluster_acc(y_true, y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
                    l_print = [np.round(float(l), 5) for l in loss_vals]
                    print(f"Res {current_res} | Ep {epoch}: ACC={acc}, NMI={nmi}, ARI={ari}, Total_L={l_print[0]}")
                # -------------------------

                # Convergence check
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if epoch > 0 and delta_label < tol:
                    print(f'Converged at resolution {current_res}')
                    break

            # Batch training
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for i in range(0, num_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                x_c = tf.cast(adata.X[batch_idx], tf.float32)
                x_s = tf.cast(adata.obs.size_factors.values[batch_idx], tf.float32)
                topo_input_batch = None
                y_p_batch = tf.cast(p[batch_idx], tf.float32)
                y_r = adata.raw.X[batch_idx] if use_raw_as_output else adata.X[batch_idx]

                if topo_input is not None:
                    topo_input_batch = tf.cast(topo_input[batch_idx], tf.float32)

                loss_vals = train_step(
                    x_c, x_s, y_p_batch, y_r, 
                    network=network, 
                    model=model, 
                    clustering_layer=clustering_layer,
                    ae_loss_fn=ae_loss_fn, 
                    opt_dec=opt_dec, 
                    loss_weights=current_weights,
                    topo_size=topo_size,
                    pg_dist=pg_dist, order=order,
                    topo_latent_mode=topo_latent_mode,
                    k=k, t=t, topo_input_batch=topo_input_batch,
                    rips_layer=rips_layer
                )

            current_loss = float(loss_vals[0])

            # --- Logic for EarlyStopping and best weights ---
            if current_loss < phase_best_loss - tol:
                phase_best_loss = current_loss
                best_weights = [tf.identity(w) for w in model.get_weights()]
                es_wait = 0
            else:
                es_wait += 1
                if es_wait >= early_stop_patience and cluster_early_stop:
                    print(f"Phase {current_res} early stopping at epoch {epoch}")
                    break

            if verbose:
                print(f"Res {current_res} Ep {epoch} - Total L: {loss_vals[0]:.4f}, "
                      f"L_zinb: {loss_vals[1]:.4f}, L_kl: {loss_vals[2]:.4f}, "
                      f"L_sk: {loss_vals[3]:.4f}, L_topo: {loss_vals[4]:.4f}")
                
    # --- Restore best weights ---
    if best_weights is not None:
        print("Restoring best weights from training history...")
        model.set_weights(best_weights)

    print(f"Ramp Training complete in {time.time() - start_total_train:.2f}s")
    
    return y_pred
