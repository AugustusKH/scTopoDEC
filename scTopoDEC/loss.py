import numpy as np
import tensorflow as tf
import keras
from keras import ops

from .utils import density_scale
import gudhi.tensorflow.rips_layer as rips
from gudhi.tensorflow.pers_dist_layer import WassersteinDistance
from gudhi.tensorflow.pers_img_layer import PersistenceImage


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

def topo_loss(x, z, homology_dim=1, maximum_edge_length=2., p=2):
    """
    Calculate topological loss between two different spaces using differentiable 
    Wasserstein distance on persistence diagrams.

    # Arguments:
    x: Input space (batch_size, feature_size). High-dimensional gene expression data.
    z: Latent space (batch_size, latent_dim). Low-dimensional manifold embedding.
    
    homology_dim: The Betti number/dimension to calculate. 
                  - 0: Connected components (clusters).
                  - 1: Cycles/loops (trajectories/branches).
                  - 2: Voids/spheres (globular structures).
                  
    maximum_edge_length: The filtration cutoff. Limits the distance at which 
                         points are connected. Prevents OOM errors by ignoring 
                         very long-distance edges.
                         
    p: The order of the Wasserstein distance (Lp norm). 
       - p=1: Earth Mover's Distance (linear penalty).
       - p=2: Standard Euclidean Wasserstein (smooth gradients).
       - Higher p: Approximates Bottleneck distance (focuses on the largest error).
    """
    # Setup Rips layers for both spaces
    rips_x = rips.RipsPersistence(homology_dimensions=[homology_dim], 
                                  maximum_edge_length=maximum_edge_length)
    rips_z = rips.RipsPersistence(homology_dimensions=[homology_dim], 
                                  maximum_edge_length=maximum_edge_length)
    
    # Sacle the space
    x_scaled = density_scale(x)
    z_scaled = density_scale(z)

    # Get persistent diagrams
    diag_x = rips_x(x_scaled)[0]
    diag_z = rips_z(z_scaled)[0]

    # Add a small dummy in case of batch is very homogenous
    if tf.shape(diag_x)[0] == 0:
        diag_x = tf.zeros((1, 2))
    if tf.shape(diag_z)[0] == 0:
        diag_z = tf.zeros((1, 2))

    # Calculate differentiable Wasserstein distane 
    pdgm_dist = WassersteinDistance(order=p)
    loss = pdgm_dist([diag_x, diag_z])

    return loss


def topo_pi_loss(x, z, homology_dim=1, maximum_edge_length=2., bandwidth=0.1, resolution=[20, 20]):
    """
    Calculate topological loss using Persistence Images (PI) and mean square error (MSE).
    
    # Arguments:
    x: Input space (batch_size, feature_size). High-dimensional gene expression data.
    z: Latent space (batch_size, latent_dim). Low-dimensional manifold embedding.
    
    homology_dim: The Betti number/dimension to calculate. 
                  - 0: Connected components (clusters).
                  - 1: Cycles/loops (trajectories/branches).
                  - 2: Voids/spheres (globular structures).
                  
    maximum_edge_length: The filtration cutoff. Limits the distance at which 
                         points are connected. Prevents OOM errors by ignoring 
                         very long-distance edges.

    bandwidth: The standard deviation of the Gaussian kernels used to spread 
               the persistence of each point across the image pixels.
    resolution: The [rows, columns] of the resulting persistence image.
    """
    # Setup Rips layers for both spaces
    rips_x = rips.RipsPersistence(homology_dimensions=[homology_dim], 
                                  maximum_edge_length=maximum_edge_length)
    rips_z = rips.RipsPersistence(homology_dimensions=[homology_dim], 
                                  maximum_edge_length=maximum_edge_length)
    
    # Sacle the space
    x_scaled = scale_by_density(x)
    z_scaled = scale_by_density(z)

    # Get persistent diagrams
    diag_x = rips_x(x_scaled)[0]
    diag_z = rips_z(z_scaled)[0]

    # Construct PIs and calculate MSE
    imager = PersistenceImage(bandwidth=bandwidth, resolution=resolution)
    pi_x = imager(diag_x)
    pi_z = imager(diag_z)
    loss = tf.reduce_mean(tf.square(pi_x - pi_z))

    return loss
