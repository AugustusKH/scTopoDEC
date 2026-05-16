import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Modernized Activations using TF2 math operations
MeanAct = lambda x: tf.clip_by_value(tf.math.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.math.softplus(x), 1e-4, 1e4)


def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)


def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x) + np.inf, x)


def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)


def NB(theta, y_true, y_pred, mask=False, debug=False, mean=False):
    eps = 1e-10
    scale_factor = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    
    if mask:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)
        
    theta = tf.minimum(theta, 1e6)
    
    # Modernized math calls replacing legacy tf.lgamma and tf.log
    t1 = tf.math.lgamma(theta + eps) + tf.math.lgamma(y_true + 1.0) - tf.math.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * tf.math.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.math.log(theta + eps) - tf.math.log(y_pred + eps)))
    
    if debug:
        # TF2 replaces verify_tensor_all_finite with explicit tf.debugging operations
        tf.debugging.assert_all_finite(y_pred, 'y_pred has inf/nans')
        tf.debugging.assert_all_finite(t1, 't1 has inf/nans')
        tf.debugging.assert_all_finite(t2, 't2 has inf/nans')
        final = t1 + t2
    else:
        final = t1 + t2
        
    final = _nan2inf(final)
    if mean:
        if mask:
            final = tf.divide(tf.reduce_sum(final), nelem)
        else:
            final = tf.reduce_mean(final)
    return final


def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mean=True, mask=False, debug=False):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_true, y_pred, mean=False, debug=debug) - tf.math.log(1.0 - pi + eps)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    theta = tf.minimum(theta, 1e6)

    zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -tf.math.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = tf.where(tf.less(y_true, tf.cast(1e-8, tf.float32)), zero_case, nb_case)
    
    ridge = ridge_lambda * tf.square(pi)
    result += ridge
    
    if mean:
        if mask:
            result = _reduce_mean(result)
        else:
            result = tf.reduce_mean(result)

    result = _nan2inf(result)
    return result


def cal_latent(hidden, alpha):
    """ComputesStudent-t similarity probability distributions over embeddings."""
    sum_y = tf.reduce_sum(tf.square(hidden), axis=1)
    num = -2.0 * tf.matmul(hidden, tf.transpose(hidden)) + tf.reshape(sum_y, [-1, 1]) + sum_y
    num = num / alpha
    num = tf.pow(1.0 + num, -(alpha + 1.0) / 2.0)
    
    # Modern matrix diagonal modifications
    diag_part = tf.linalg.diag_part(num)
    zerodiag_num = num - tf.linalg.diag(diag_part)
    
    latent_p = tf.transpose(tf.transpose(zerodiag_num) / tf.reduce_sum(zerodiag_num, axis=1))
    return num, latent_p


def target_dis(latent_p):
    """Computes the target distribution P to minimize KL divergence optimization loops."""
    latent_q = tf.transpose(tf.transpose(tf.pow(latent_p, 2)) / tf.reduce_sum(latent_p, axis=1))
    return tf.transpose(tf.transpose(latent_q) / tf.reduce_sum(latent_q, axis=1))


def cal_dist(hidden, clusters):
    """Calculates cluster similarities using squared soft assignments."""
    dist1 = tf.reduce_sum(tf.square(tf.expand_dims(hidden, axis=1) - clusters), axis=2)
    temp_dist1 = dist1 - tf.reshape(tf.reduce_min(dist1, axis=1), [-1, 1])
    q = tf.math.exp(-temp_dist1)
    q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
    q = tf.pow(q, 2)
    q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
    dist2 = dist1 * q
    return dist1, dist2