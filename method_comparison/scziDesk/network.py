import logging
import os
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

# Assuming local loss utilities are updated to the TF2 versions we modified
from loss import NB, ZINB, cal_latent, target_dis, cal_dist, DispAct, MeanAct

logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Unbounded Gaussian Noise Layer (Bypasses Modern Keras Constraints)
# ==============================================================================
class UnboundedGaussianNoise(tf.keras.layers.Layer):
    """
    Applies additive Gaussian noise to the inputs during training mode.
    Bypasses modern Keras restriction that caps standard deviation scaling.
    """
    def __init__(self, stddev, name=None, **kwargs):
        super(UnboundedGaussianNoise, self).__init__(name=name, **kwargs)
        self.stddev = stddev

    def call(self, inputs, training=False):
        if training:
            # Inject native normal distribution noise matched to the target shape
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev, dtype=inputs.dtype)
            return inputs + noise
        return inputs


# ==============================================================================
# Modernized TF2 scziDesk Autoencoder Class
# ==============================================================================
class ScziDeskAutoencoder(tf.keras.Model):
    """
    Unified TensorFlow 2.x Deep Embedding Clustering network for scziDesk benchmarks.
    """
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
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.init = init
        self.act = act
        self.n_stacks = len(self.dims) - 1

        # Modernized Optimizer tracking state
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Cluster Centroid Weights Variable
        self.clusters = tf.Variable(
            initial_value=tf.keras.initializers.get(self.init)(shape=[self.cluster_num, self.dims[-1]]),
            name=f"{self.dataname}/clusters_rep",
            trainable=True,
            dtype=tf.float32
        )

        # 1. Structural Encoder Block Architecture (FIXED: Using Custom Unbounded Noise)
        self.encoder_noise_in = UnboundedGaussianNoise(self.noise_sd, name='input_noise')
        self.encoder_layers = []
        self.encoder_noise_layers = []
        self.encoder_acts = []
        
        for i in range(self.n_stacks - 1):
            self.encoder_layers.append(
                tf.keras.layers.Dense(units=self.dims[i + 1], kernel_initializer=self.init, name=f'encoder_{i}')
            )
            # FIXED: Swapped out legacy layer to prevent value range crashes
            self.encoder_noise_layers.append(
                UnboundedGaussianNoise(self.noise_sd, name=f'noise_{i}')
            )
            self.encoder_acts.append(
                tf.keras.layers.Activation(self.act)
            )
            
        self.latent_layer = tf.keras.layers.Dense(units=self.dims[-1], kernel_initializer=self.init, name='encoder_hidden')

        # 2. Structural Decoder Block Architecture
        self.decoder_layers = []
        for i in range(self.n_stacks - 1, 0, -1):
            self.decoder_layers.append(
                tf.keras.layers.Dense(units=self.dims[i], activation=self.act, kernel_initializer=self.init, name=f'decoder_{i}')
            )

        # 3. Parameter Mapping Output Heads
        if self.distribution == "ZINB":
            self.pi_layer = tf.keras.layers.Dense(units=self.dims[0], activation='sigmoid', kernel_initializer=self.init, name='pi')
            self.disp_layer = tf.keras.layers.Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name='dispersion')
            self.mean_layer = tf.keras.layers.Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')
        elif self.distribution == "NB":
            self.disp_layer = tf.keras.layers.Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name='dispersion')
            self.mean_layer = tf.keras.layers.Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')

    def encode(self, x, training=False):
        h = self.encoder_noise_in(x, training=training)
        for i in range(self.n_stacks - 1):
            h = self.encoder_layers[i](h)
            h = self.encoder_noise_layers[i](h, training=training)
            h = self.encoder_acts[i](h)
        return self.latent_layer(h)

    def decode(self, latent):
        h = latent
        for layer in self.decoder_layers:
            h = layer(h)
        return h

    def call(self, inputs, training=False):
        x, sf_layer = inputs
        latent = self.encode(x, training=training)
        h_decoded = self.decode(latent)

        if self.distribution == "ZINB":
            pi = self.pi_layer(h_decoded)
            disp = self.disp_layer(h_decoded)
            mean = self.mean_layer(h_decoded)
            output = mean * tf.matmul(sf_layer, tf.ones((1, mean.shape[1]), dtype=tf.float32))
            return latent, pi, disp, output
        elif self.distribution == "NB":
            disp = self.disp_layer(h_decoded)
            mean = self.mean_layer(h_decoded)
            output = mean * tf.matmul(sf_layer, tf.ones((1, mean.shape[1]), dtype=tf.float32))
            return latent, disp, output

    @tf.function
    def train_step_pretrain(self, x, x_count, sf_layer):
        """Compiles efficient isolated pretraining gradients using tape tracking."""
        with tf.GradientTape() as tape:
            if self.distribution == "ZINB":
                latent, pi, disp, output = self((x, sf_layer), training=True)
                likelihood_loss = ZINB(pi, disp, x_count, output, ridge_lambda=1.0)
            elif self.distribution == "NB":
                latent, disp, output = self((x, sf_layer), training=True)
                likelihood_loss = NB(disp, x_count, output, mask=False, debug=False, mean=True)
                
        gradients = tape.gradient(likelihood_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return likelihood_loss, latent

    @tf.function
    def train_step_finetune(self, x, x_count, sf_layer):
        """Executes full multi-objective loss corrections across joint variables."""
        with tf.GradientTape() as tape:
            if self.distribution == "ZINB":
                latent, pi, disp, output = self((x, sf_layer), training=True)
                likelihood_loss = ZINB(pi, disp, x_count, output, ridge_lambda=1.0)
            elif self.distribution == "NB":
                latent, disp, output = self((x, sf_layer), training=True)
                likelihood_loss = NB(disp, x_count, output, mask=False, debug=False, mean=True)

            # Re-verify neighborhood clusters calculations from loss module
            num, latent_p = cal_latent(latent, self.t_alpha)
            latent_q = target_dis(latent_p)
            
            diag_mask = tf.linalg.diag(tf.linalg.diag_part(num))
            latent_p_masked = latent_p + diag_mask
            latent_q_masked = latent_q + diag_mask
            
            latent_dist1, latent_dist2 = cal_dist(latent, self.clusters)
            kmeans_loss = tf.reduce_mean(tf.reduce_sum(latent_dist2, axis=1))

            if self.self_training:
                cross_entropy = -tf.reduce_sum(latent_q_masked * tf.math.log(latent_p_masked))
                entropy = -tf.reduce_sum(latent_q_masked * tf.math.log(latent_q_masked))
                kl_loss = cross_entropy - entropy
                total_loss = likelihood_loss + self.alpha * kmeans_loss + self.gamma * kl_loss
            else:
                total_loss = likelihood_loss + self.alpha * kmeans_loss
                
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return total_loss, latent_dist1

    def pretrain(self, X, count_X, size_factor, batch_size, pretrain_epoch, gpu_option):
        print("begin the pretraining")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option
        
        # Safe device allocation management
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.error(f"Memory growth configuration error: {e}")

        self.latent_repre = np.zeros((X.shape[0], self.dims[-1]))
        num_samples = X.shape[0]

        for ite in range(pretrain_epoch):
            indices = np.arange(num_samples)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_idx = indices[start_idx:end_idx]
                
                # Handle cyclic mini-batch padding constraints natively
                if len(batch_idx) < batch_size:
                    padding_needed = batch_size - len(batch_idx)
                    batch_idx = np.concatenate([batch_idx, indices[:padding_needed]])

                x_batch = tf.convert_to_tensor(X[batch_idx], dtype=tf.float32)
                x_count_batch = tf.convert_to_tensor(count_X[batch_idx], dtype=tf.float32)
                sf_batch = tf.convert_to_tensor(size_factor[batch_idx], dtype=tf.float32)

                loss_val, latent_val = self.train_step_pretrain(x_batch, x_count_batch, sf_batch)
                self.latent_repre[batch_idx[:end_idx-start_idx]] = latent_val.numpy()[:end_idx-start_idx]

    def funetrain(self, X, count_X, size_factor, batch_size, funetrain_epoch, update_epoch, error):
        kmeans = KMeans(n_clusters=self.cluster_num, init="k-means++", n_init=10)
        self.latent_repre = np.nan_to_num(self.latent_repre)
        self.kmeans_pred = kmeans.fit_predict(self.latent_repre)
        self.last_pred = np.copy(self.kmeans_pred)
        
        # Directly update cluster variables without tf.assign session handles
        self.clusters.assign(kmeans.cluster_centers_)
        print("begin the funetraining")

        num_samples = X.shape[0]
        for i in range(1, funetrain_epoch + 1):
            if i % update_epoch == 0:
                # Direct full evaluation forwarding passes
                x_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
                sf_tensor = tf.convert_to_tensor(size_factor, dtype=tf.float32)
                
                latent_all = self.encode(x_tensor, training=False)
                dist, _ = cal_dist(latent_all, self.clusters)
                
                self.Y_pred = np.argmin(dist.numpy(), axis=1)
                if np.sum(self.Y_pred != self.last_pred) / len(self.last_pred) < error:
                    break
                else:
                    self.last_pred = self.Y_pred
            else:
                indices = np.arange(num_samples)
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_idx = indices[start_idx:end_idx]
                    
                    if len(batch_idx) < batch_size:
                        padding_needed = batch_size - len(batch_idx)
                        batch_idx = np.concatenate([batch_idx, indices[:padding_needed]])

                    x_batch = tf.convert_to_tensor(X[batch_idx], dtype=tf.float32)
                    x_count_batch = tf.convert_to_tensor(count_X[batch_idx], dtype=tf.float32)
                    sf_batch = tf.convert_to_tensor(size_factor[batch_idx], dtype=tf.float32)

                    self.train_step_finetune(x_batch, x_count_batch, sf_batch)
                    
        return self.Y_pred