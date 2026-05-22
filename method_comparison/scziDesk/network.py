import os
import logging
import numpy as np
import tensorflow as tf
from keras import layers, initializers, optimizers
from sklearn.cluster import KMeans
from loss import NB, ZINB, cal_latent, target_dis, cal_dist, DispAct, MeanAct

logger = logging.getLogger(__name__)

class ScziDeskAutoencoder(tf.keras.Model):
    def __init__(self, dataname, distribution, self_training, dims, cluster_num, 
                 t_alpha, alpha, gamma, learning_rate, noise_sd=1.5, init='glorot_uniform', act='relu'):
        super(ScziDeskAutoencoder, self).__init__()
        self.dataname = dataname
        self.distribution = distribution
        self.self_training = self_training
        self.dims = dims
        self.cluster_num = cluster_num
        self.t_alpha = t_alpha
        self.alpha = alpha
        self.gamma = gamma
        self.noise_sd = noise_sd
        self.n_stacks = len(self.dims) - 1

        # High-precision epsilon for optimizer stability
        self.optimizer = optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)

        # Cluster Centers
        self.clusters = tf.Variable(
            initial_value=initializers.get(init)(shape=[self.cluster_num, self.dims[-1]]),
            name="clusters_rep", trainable=True, dtype=tf.float32
        )

        # Encoder Architecture
        self.encoder_noise_in = layers.GaussianNoise(noise_sd)
        self.encoder_layers = []
        self.encoder_noise_layers = []
        self.encoder_acts = []
        
        for i in range(self.n_stacks - 1):
            self.encoder_layers.append(layers.Dense(units=self.dims[i + 1], kernel_initializer=init))
            self.encoder_noise_layers.append(layers.GaussianNoise(noise_sd))
            self.encoder_acts.append(layers.Activation(act))
            
        self.latent_layer = layers.Dense(units=self.dims[-1], kernel_initializer=init, name='encoder_hidden')

        # Decoder Architecture
        self.decoder_layers = []
        for i in range(self.n_stacks - 1, 0, -1):
            self.decoder_layers.append(layers.Dense(units=self.dims[i], activation=act, kernel_initializer=init))

        # Output Heads
        if self.distribution == "ZINB":
            self.pi_layer = layers.Dense(units=self.dims[0], activation='sigmoid', kernel_initializer=init)
        self.disp_layer = layers.Dense(units=self.dims[0], activation=DispAct, kernel_initializer=init)
        self.mean_layer = layers.Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=init)

    def encode(self, x, training=False):
        h = self.encoder_noise_in(x, training=training)
        for i in range(self.n_stacks - 1):
            h = self.encoder_layers[i](h)
            h = self.encoder_noise_layers[i](h, training=training)
            h = self.encoder_acts[i](h)
        return self.latent_layer(h)

    def call(self, inputs, training=False):
        x, sf = inputs
        latent = self.encode(x, training=training)
        h = latent
        for layer in self.decoder_layers:
            h = layer(h)
        
        if self.distribution == "ZINB":
            pi, disp, mean = self.pi_layer(h), self.disp_layer(h), self.mean_layer(h)
            return latent, pi, disp, mean * sf
        else:
            disp, mean = self.disp_layer(h), self.mean_layer(h)
            return latent, disp, mean * sf

    @tf.function
    def train_step_pretrain(self, x, x_count, sf):
        """Isolated pretraining: Only optimizes the autoencoder reconstruction loss."""
        with tf.GradientTape() as tape:
            if self.distribution == "ZINB":
                latent, pi, disp, output = self((x, sf), training=True)
                loss = ZINB(pi, disp, x_count, output, ridge_lambda=1.0)
            else:
                latent, disp, output = self((x, sf), training=True)
                loss = NB(disp, x_count, output, mask=False, mean=True)
                
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss, latent

    @tf.function
    def train_step_finetune(self, x, x_count, sf):
        """Joint finetuning: Optimizes reconstruction + K-means clustering + KL divergence."""
        with tf.GradientTape() as tape:
            if self.distribution == "ZINB":
                latent, pi, disp, output = self((x, sf), training=True)
                loss = ZINB(pi, disp, x_count, output, ridge_lambda=1.0)
            else:
                latent, disp, output = self((x, sf), training=True)
                loss = NB(disp, x_count, output, mask=False, mean=True)

            # Topological/Clustering Loss
            _, latent_p = cal_latent(latent, self.t_alpha)
            latent_q = target_dis(latent_p)
            
            latent_p = tf.clip_by_value(latent_p, 1e-8, 1.0)
            latent_q = tf.clip_by_value(latent_q, 1e-8, 1.0)
            
            _, latent_dist2 = cal_dist(latent, self.clusters)
            kmeans_loss = tf.reduce_mean(tf.reduce_sum(latent_dist2, axis=1))
            
            total_loss = loss + self.alpha * kmeans_loss
            if self.self_training:
                kl = -tf.reduce_sum(latent_q * tf.math.log(latent_p)) - (-tf.reduce_sum(latent_q * tf.math.log(latent_q)))
                total_loss += self.gamma * kl
                
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return total_loss

    def pretrain(self, X, count_X, size_factor, batch_size, pretrain_epoch, gpu_option):
        print("Begin pretraining...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.error(f"GPU config error: {e}")

        self.latent_repre = np.zeros((X.shape[0], self.dims[-1]), dtype=np.float32)
        num_samples = X.shape[0]
        indices = np.arange(num_samples)

        for epoch in range(pretrain_epoch):
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_idx = indices[start_idx:end_idx]
                
                # Cyclic padding for the last batch
                if len(batch_idx) < batch_size:
                    padding = batch_size - len(batch_idx)
                    batch_idx = np.concatenate([batch_idx, indices[:padding]])

                x_batch = tf.convert_to_tensor(X[batch_idx], dtype=tf.float32)
                count_batch = tf.convert_to_tensor(count_X[batch_idx], dtype=tf.float32)
                sf_batch = tf.convert_to_tensor(size_factor[batch_idx], dtype=tf.float32)

                _, latent_val = self.train_step_pretrain(x_batch, count_batch, sf_batch)
                
                # Only store the actual (non-padded) items in the latent representation
                actual_len = end_idx - start_idx
                self.latent_repre[start_idx:end_idx] = latent_val.numpy()[:actual_len]

    def funetrain(self, X, count_X, size_factor, batch_size, funetrain_epoch, update_epoch, error):
        print("Begin finetuning...")
        kmeans = KMeans(n_clusters=self.cluster_num, init="k-means++", n_init=10)
        
        # Initial K-Means assignment
        self.latent_repre = np.nan_to_num(self.latent_repre)
        self.kmeans_pred = kmeans.fit_predict(self.latent_repre)
        self.last_pred = np.copy(self.kmeans_pred)
        self.clusters.assign(kmeans.cluster_centers_)

        num_samples = X.shape[0]
        indices = np.arange(num_samples)

        for epoch in range(1, funetrain_epoch + 1):
            if epoch % update_epoch == 0:
                # Full evaluation pass to update predictions and check convergence
                x_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
                sf_tensor = tf.convert_to_tensor(size_factor, dtype=tf.float32)
                
                latent_all = self.encode(x_tensor, training=False)
                dist, _ = cal_dist(latent_all, self.clusters)
                
                self.Y_pred = np.argmin(dist.numpy(), axis=1)
                delta_label = np.sum(self.Y_pred != self.last_pred).astype(np.float32) / self.Y_pred.shape[0]
                
                if delta_label < error:
                    print(f"Reached tolerance threshold at epoch {epoch}. Stopping training.")
                    break
                else:
                    self.last_pred = np.copy(self.Y_pred)
            else:
                # Standard mini-batch training loop
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_idx = indices[start_idx:end_idx]
                    
                    if len(batch_idx) < batch_size:
                        padding = batch_size - len(batch_idx)
                        batch_idx = np.concatenate([batch_idx, indices[:padding]])

                    x_batch = tf.convert_to_tensor(X[batch_idx], dtype=tf.float32)
                    count_batch = tf.convert_to_tensor(count_X[batch_idx], dtype=tf.float32)
                    sf_batch = tf.convert_to_tensor(size_factor[batch_idx], dtype=tf.float32)

                    self.train_step_finetune(x_batch, count_batch, sf_batch)
                    
        return self.Y_pred