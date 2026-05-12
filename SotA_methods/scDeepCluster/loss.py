import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)


def mse_loss(y_true, y_pred):
    ret = tf.square(y_pred - y_true)
    return _reduce_mean(ret)


def poisson_loss(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    nelem = _nelem(y_true)
    y_true = _nan2zero(y_true)

    # Replaced tf.log and tf.lgamma with modern tf.math variants
    ret = y_pred - y_true * tf.math.log(y_pred + 1e-10) + tf.math.lgamma(y_true + 1.0)

    return tf.divide(tf.reduce_sum(ret), nelem)


class NB(object):
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False):

        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            if self.masking:
                nelem = _nelem(y_true)
                y_true = _nan2zero(y_true)

            # Clip theta
            theta = tf.minimum(self.theta, 1e6)

            # Replaced tf.log and tf.lgamma with modern tf.math variants
            t1 = tf.math.lgamma(theta+eps) + tf.math.lgamma(y_true+1.0) - tf.math.lgamma(y_true+theta+eps)
            t2 = (theta+y_true) * tf.math.log(1.0 + (y_pred/(theta+eps))) + (y_true * (tf.math.log(theta+eps) - tf.math.log(y_pred+eps)))

            if self.debug:
                # Replaced deprecated verify_tensor_all_finite with modern assert_all_finite
                assert_ops = [
                        tf.debugging.assert_all_finite(y_pred, 'y_pred has inf/nans'),
                        tf.debugging.assert_all_finite(t1, 't1 has inf/nans'),
                        tf.debugging.assert_all_finite(t2, 't2 has inf/nans')]

                # TF2 handles summaries differently; disabled inline summaries inside loss to prevent graph crashes
                # tf.summary.histogram('t1', t1)
                # tf.summary.histogram('t2', t2)

                with tf.control_dependencies(assert_ops):
                    final = t1 + t2

            else:
                final = t1 + t2

            final = _nan2inf(final)

            if mean:
                if self.masking:
                    final = tf.divide(tf.reduce_sum(final), nelem)
                else:
                    final = tf.reduce_mean(final)

        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            # reuse existing NB neg.log.lik.
            nb_case = super().loss(y_true, y_pred, mean=False) - tf.math.log(1.0-self.pi+eps)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor
            theta = tf.minimum(self.theta, 1e6)

            zero_nb = tf.pow(theta/(theta+y_pred+eps), theta)
            zero_case = -tf.math.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
            result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
            ridge = self.ridge_lambda*tf.square(self.pi)
            result += ridge

            if mean:
                if self.masking:
                    result = _reduce_mean(result)
                else:
                    result = tf.reduce_mean(result)

            result = _nan2inf(result)

            if self.debug:
                # Disabled inline summaries for TF2 eager/graph safety inside loss functions
                pass
                # tf.summary.histogram('nb_case', nb_case)
                # tf.summary.histogram('zero_nb', zero_nb)
                # tf.summary.histogram('zero_case', zero_case)
                # tf.summary.histogram('ridge', ridge)

        return result