"""
Implementation of scDeepCluster for scRNA-seq data (Modernized for TF2/Python 3 with Batch Correction)
"""

from time import time
import numpy as np
import h5py
import csv
import os

# Modern TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec, Dense, Input, GaussianNoise, Activation, Concatenate
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment 

import scanpy as sc 

# Import from your local updated files
from layers import ConstantDispersionLayer, SliceLayer, ColWiseMultLayer
from loss import poisson_loss, NB, ZINB
from preprocess import read_dataset, normalize

# Modern random seed setting
np.random.seed(2211)
tf.random.set_seed(2211)

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def build_autoencoder(dims, n_batch, noise_sd=0, init='glorot_uniform', act='relu'):
    """
    Fully connected auto-encoder model, conditional on batch labels.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
        n_batch: dimensionality of the batch one-hot vector.
        act: activation, not applied to Input, Hidden and Output layers
    return:
        Tuple of (encoder_model, autoencoder_model)
    """
    # Inputs
    sf_layer = Input(shape=(1,), name='size_factors')
    x = Input(shape=(dims[0],), name='counts')
    b = Input(shape=(n_batch,), name='batch')

    # Encoder
    h = x
    h = GaussianNoise(noise_sd, name='input_noise')(h)
    h = Concatenate(name='concat_encoder_input')([h, b]) # Condition encoder on batch
    
    for i in range(len(dims) - 2):
        h = Dense(dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h)
        h = GaussianNoise(noise_sd, name='noise_%d' % i)(h)    
        h = Activation(act)(h)
        
    # Latent space (bottleneck)
    latent = Dense(dims[-1], kernel_initializer=init, name='encoder_hidden')(h)  

    # Define the isolated encoder model
    encoder = Model(inputs=[x, b], outputs=latent, name='encoder')

    # Decoder
    h_dec = Concatenate(name='concat_decoder_input')([latent, b]) # Condition decoder on batch
    
    for i in range(len(dims) - 2, 0, -1):
        h_dec = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(h_dec)

    # Output parameters for ZINB
    pi = Dense(dims[0], activation='sigmoid', kernel_initializer=init, name='pi')(h_dec)
    disp = Dense(dims[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h_dec)
    mean = Dense(dims[0], activation=MeanAct, kernel_initializer=init, name='mean')(h_dec)

    output = ColWiseMultLayer(name='output')([mean, sf_layer])
    output = SliceLayer(0, name='slice')([output, disp, pi])

    # Define full autoencoder
    autoencoder = Model(inputs=[x, sf_layer, b], outputs=output, name='autoencoder')

    return encoder, autoencoder


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    """
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SCDeepClusterBatch(object):
    def __init__(self, dims, n_batch, n_clusters=10, noise_sd=0, alpha=1.0, ridge=0, debug=False):
        super(SCDeepCluster, self).__init__()
        self.dims = dims
        self.input_dim = dims[0]
        self.n_batch = n_batch
        self.n_stacks = len(self.dims) - 1
        self.n_clusters = n_clusters
        self.noise_sd = noise_sd
        self.alpha = alpha
        self.act = 'relu'
        self.ridge = ridge
        self.debug = debug
        
        # Build encoder and autoencoder models cleanly
        self.encoder, self.autoencoder = build_autoencoder(self.dims, self.n_batch, noise_sd=self.noise_sd, act=self.act)
        
        # Isolate ZINB parameters for loss function
        pi = self.autoencoder.get_layer(name='pi').output
        disp = self.autoencoder.get_layer(name='dispersion').output
        mean = self.autoencoder.get_layer(name='mean').output
        zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
        self.loss = zinb.loss

        # Connect clustering layer to the latent bottleneck output
        latent = self.encoder.output
        clustering_layer = ClusteringLayer(self.n_clusters, alpha=self.alpha, name='clustering')(latent)
        
        # Compile final multi-output model
        self.model = Model(inputs=self.autoencoder.input, # [x, sf_layer, b]
                           outputs=[clustering_layer, self.autoencoder.output])

        self.pretrained = False
        self.centers = []
        self.y_pred = []

    def pretrain(self, x_counts, sf, b, y_raw, batch_size=256, epochs=200, optimizer='adam', ae_file='ae_weights.h5'):
        print('...Pretraining autoencoder...')
        self.autoencoder.compile(loss=self.loss, optimizer=optimizer)
        es = EarlyStopping(monitor="loss", patience=50, verbose=1)
        self.autoencoder.fit(x=[x_counts, sf, b], y=y_raw, batch_size=batch_size, epochs=epochs, callbacks=[es])
        self.autoencoder.save_weights(ae_file)
        print('Pretrained weights are saved to ./' + str(ae_file))
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x_counts, b):
        return self.encoder.predict([x_counts, b])

    def predict_clusters(self, x_counts, sf, b):
        q, _ = self.model.predict([x_counts, sf, b], verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def fit(self, x_counts, sf, b, y, raw_counts, batch_size=256, maxiter=2e4, tol=1e-3, update_interval=140,
            ae_weights=None, save_dir='./results/scDeepCluster', loss_weights=[1,1], optimizer='adadelta'):

        self.model.compile(loss=['kld', self.loss], loss_weights=loss_weights, optimizer=optimizer)
        print('Update interval', update_interval)
        save_interval = int(x_counts.shape[0] / batch_size) * 5
        print('Save interval', save_interval)

        # Step 1: pretrain
        if not self.pretrained and ae_weights is None:
            print('...pretraining autoencoders using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x_counts=x_counts, sf=sf, b=b, y_raw=raw_counts, batch_size=batch_size)
            self.pretrained = True
        elif ae_weights is not None:
            self.autoencoder.load_weights(ae_weights)
            print('ae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means on the latent representation
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict([x_counts, b]))
        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/scDeepCluster_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict([x_counts, sf, b], verbose=0)
                p = self.target_distribution(q) 

                self.y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(cluster_acc(y, self.y_pred), 5)
                    nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    loss = np.round(loss, 5)
                    logwriter.writerow(dict(iter=ite, acc=acc, nmi=nmi, ari=ari, L=loss[0], Lc=loss[1], Lr=loss[2]))
                    print('Iter-%d: ACC= %.4f, NMI= %.4f, ARI= %.4f;  L= %.5f, Lc= %.5f,  Lr= %.5f'
                          % (ite, acc, nmi, ari, loss[0], loss[1], loss[2]))

                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x_counts.shape[0]:
                loss = self.model.train_on_batch(
                    x=[x_counts[index * batch_size::], sf[index * batch_size:], b[index * batch_size::]],
                    y=[p[index * batch_size::], raw_counts[index * batch_size::]]
                )
                index = 0
            else:
                loss = self.model.train_on_batch(
                    x=[x_counts[index * batch_size:(index + 1) * batch_size], 
                       sf[index * batch_size:(index + 1) * batch_size],
                       b[index * batch_size:(index + 1) * batch_size]],
                    y=[p[index * batch_size:(index + 1) * batch_size], 
                       raw_counts[index * batch_size:(index + 1) * batch_size]]
                )
                index += 1

            if ite % save_interval == 0:
                print('saving model to: ' + save_dir + '/scDeepCluster_model_' + str(ite) + '.h5')
                self.model.save_weights(save_dir + '/scDeepCluster_model_' + str(ite) + '.h5')

        logfile.close()
        print('saving model to: ' + save_dir + '/scDeepCluster_model_final.h5')
        self.model.save_weights(save_dir + '/scDeepCluster_model_final.h5')
        
        return self.y_pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--gamma', default=1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=0, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scDeepCluster')
    parser.add_argument('--ae_weight_file', default='ae_weights.h5')

    args = parser.parse_args()

    # load dataset
    optimizer1 = Adam(amsgrad=True)
    optimizer2 = 'adadelta'

    # Read data including batch matrix B
    data_mat = h5py.File(args.data_file, 'r')
    x = np.array(data_mat['X']).astype('float64')
    b = np.array(data_mat['B']).astype('float64')
    y = np.array(data_mat['Y'])
    data_mat.close()

    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x, dtype="float64")
    adata.obs['Group'] = y

    adata = read_dataset(adata, transpose=False, test_split=False, copy=True)
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)

    input_size = adata.n_vars
    n_batch = b.shape[1]

    print(adata.X.shape)
    print(y.shape)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

    if args.update_interval == 0:  
        args.update_interval = int(adata.X.shape[0]/args.batch_size)
    print(args)

    # Define scDeepCluster model with n_batch integrated
    scDeepCluster = SCDeepClusterBatch(dims=[input_size, 256, 64, 32], n_batch=n_batch, n_clusters=args.n_clusters, noise_sd=2.5)
    plot_model(scDeepCluster.model, to_file='scDeepCluster_model.png', show_shapes=True)
    print("autocoder summary")
    scDeepCluster.autoencoder.summary()
    print("model summary")
    scDeepCluster.model.summary()

    t0 = time()

    # Pretrain autoencoders before clustering
    if args.ae_weights is None:
        scDeepCluster.pretrain(x_counts=adata.X, sf=adata.obs['size_factors'].values, b=b, y_raw=adata.raw.X, batch_size=args.batch_size, epochs=args.pretrain_epochs, optimizer=optimizer1, ae_file=args.ae_weight_file)

    # begin clustering
    scDeepCluster.fit(x_counts=adata.X, sf=adata.obs['size_factors'].values, b=b, y=y, raw_counts=adata.raw.X, batch_size=args.batch_size, tol=args.tol, maxiter=args.maxiter,
             update_interval=args.update_interval, ae_weights=args.ae_weights, save_dir=args.save_dir, loss_weights=[args.gamma, 1], optimizer=optimizer2)

    # Show the final results
    y_pred = scDeepCluster.y_pred
    acc = np.round(cluster_acc(y, scDeepCluster.y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, scDeepCluster.y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, scDeepCluster.y_pred), 5)
    print('Final: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
    print('Clustering time: %d seconds.' % int(time() - t0))