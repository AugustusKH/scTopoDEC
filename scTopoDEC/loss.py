import numpy as np
import tensorflow as tf
import keras
from keras import ops

from .utils import density_scale
import gudhi as gd
from gudhi.tensorflow import RipsLayer



def _nan2zero(x):
    return ops.where(ops.isnan(x), ops.zeros_like(x), x)

def _nan2inf(x):
    return ops.where(ops.isnan(x), ops.full(ops.shape(x), np.inf, dtype=ops.dtype(x)), x)

def _nelem(x):
    mask = ops.cast(ops.logical_not(ops.isnan(x)), "float32")
    nelem = ops.sum(mask)
    return ops.cast(ops.where(ops.equal(nelem, 0.), 1., nelem), ops.dtype(x))

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return ops.sum(x) / nelem

def mse_loss(y_true, y_pred):
    ret = ops.square(y_pred - y_true)
    return _reduce_mean(ret)


# We need a class (or closure) here,
# because it's not possible to
# pass extra arguments to Keras loss functions
# See https://github.com/fchollet/keras/issues/2121

# dispersion (theta) parameter is a scalar by default.
# scale_factor scales the nbinom mean before the
# calculation of the loss to balance the
# learning rates of theta and network weights

class NB:
    def __init__(self, theta=None, masking=False, scale_factor=1.0, debug=False):
        self.eps = 1e-10 # for numerical stability
        self.scale_factor = scale_factor
        self.debug = debug
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32") * self.scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

        theta = ops.minimum(self.theta, 1e6) # Clip theta

        t1 = (ops.cast(tf.math.lgamma(theta + self.eps), "float32") + 
              ops.cast(tf.math.lgamma(y_true + 1.0), "float32") - 
              ops.cast(tf.math.lgamma(y_true + theta + self.eps), "float32"))
        t2 = (theta + y_true) * ops.log(1.0 + (y_pred / (theta + self.eps))) + \
             (y_true * (ops.log(theta + self.eps) - ops.log(y_pred + self.eps)))

        final = t1 + t2
        final = _nan2inf(final)

        if mean:
            return ops.sum(final) / nelem if self.masking else ops.mean(final)
        
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        # NB Case: Probability the zero is NOT a dropout
        nb_case = super().loss(y_true, y_pred, mean=False) - ops.log(1.0 - self.pi + self.eps)

        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32") * self.scale_factor
        theta = ops.minimum(self.theta, 1e6)

        # Zero Case: Probability the zero IS a dropout + Probability it's a bio-zero
        zero_nb = ops.power(theta / (theta + y_pred + self.eps), theta)
        zero_case = -ops.log(self.pi + ((1.0 - self.pi) * zero_nb) + self.eps)

        # Switch between cases
        result = ops.where(ops.less(y_true, 1e-8), zero_case, nb_case)

        if self.ridge_lambda > 0:
            result += self.ridge_lambda * ops.square(self.pi)

        if mean:
            result = _reduce_mean(result) if self.masking else ops.mean(result)

        return _nan2inf(result)


def soft_kmeans_loss(z, mu):
    """
    Calculate soft k-mean clustering loss.
    # Arguments
    z: Latent representations (batch_size, latent_dim)
    mu: Cluster centroid (n_clusters, latent_dim)
    # Return
        loss scalar
    """
    # Calculate lambda prime
    diff = ops.expand_dims(z, axis=1) - ops.expand_dims(mu, axis=0)
    dist_sq = ops.sum(ops.square(diff), axis=-1)
    dist_min = ops.reshape(ops.min(dist_sq, axis=1), [-1, 1])
    temp_dist = dist_sq - dist_min # Subtract minimum distance to prevent underflow
    lambda_prime = ops.exp(-temp_dist) + 1e-12
    lambda_prime = lambda_prime / ops.sum(lambda_prime, axis=1, keepdims=True)
    
    # Calculate lambda 
    lambda_ij = ops.square(lambda_prime) + 1e-12
    lambda_ij = lambda_ij / (ops.sum(lambda_ij, axis=1, keepdims=True) + 1e-12)

    # Calculate nu
    nu_numerator = ops.matmul(ops.transpose(lambda_ij), z) # (clusters, batch) * (batch, dim) -> (clusters, dim)
    nu_denominator = ops.expand_dims(ops.sum(lambda_ij, axis=0), axis=1) + 1e-12 # (clusters, 1)
    nu_j = nu_numerator / (nu_denominator + 1e-8) # (clusters, dim)

    # Calculate Lsk
    diff_to_nu = ops.expand_dims(z, axis=1) - ops.expand_dims(nu_j, axis=0)
    dist_sq_to_nu = ops.sum(ops.square(diff_to_nu), axis=-1)
    batch_size = ops.cast(ops.shape(z)[0], "float32")   
    loss_sk = ops.sum(lambda_ij * dist_sq_to_nu) / batch_size

    return loss_sk


def topo_loss(x, z, rips_layer):
    """
    Calculate topological loss between two different spaces using
    the persistence maximization logic to a manifold preservation loss.

    # Arguments:
    x: Input space (batch_size, feature_size). High-dimensional gene expression data.
    z: Latent space (batch_size, latent_dim). Low-dimensional manifold embedding.
    rips_layer: TensorFlow layer for computing Rips persistence out of a point cloud
    """
    # Scale the spaces
    x_scaled = density_scale(x)
    z_scaled = density_scale(z)

    # Get persistent diagrams
    dgm_x = rips_layer.call(x_scaled)[0][0]
    dgm_z = rips_layer.call(z_scaled)[0][0]

    # Calculate persistence (death - birth)
    pers_x = 0.5 * (dgm_x[:, 1] - dgm_x[:, 0])
    pers_z = 0.5 * (dgm_z[:, 1] - dgm_z[:, 0])

    # Dynamic padding to match shapes
    n_x = tf.shape(pers_x)[0]
    n_z = tf.shape(pers_z)[0]
    max_features = tf.reduce_max([n_x, n_z])
    pers_x_padded = tf.pad(pers_x, [[0, max_features - n_x]])
    pers_z_padded = tf.pad(pers_z, [[0, max_features - n_z]])
    pers_x_padded = tf.reshape(pers_x_padded, [max_features])
    pers_z_padded = tf.reshape(pers_z_padded, [max_features])

    # Sort vectors descending to align the most prominent features
    pers_x_sorted = tf.sort(pers_x_padded, direction='DESCENDING')
    pers_z_sorted = tf.sort(pers_z_padded, direction='DESCENDING')

    # Calculate MSE on aligned vectors
    sq_diff = tf.square(pers_x_sorted - pers_z_sorted)
    loss = tf.reduce_sum(sq_diff) / (tf.cast(max_features, tf.float32) + 1e-8)

    return loss
