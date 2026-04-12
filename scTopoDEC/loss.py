import numpy as np
import tensorflow as tf
import keras
from keras import ops


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

