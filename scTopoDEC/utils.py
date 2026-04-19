import numpy as np
import tensorflow as tf
from keras import ops


def compute_target_distribution(q):
    """
    Compute the target distribution p from soft labels q.
    p is designed to sharpen the clusters and emphasize high-confidence assignments.
    """
    q = np.clip(q, 1e-10, 1.0)
    weight = ops.power(q, 2) / (ops.sum(q, axis=0) + 1e-10) # Add 1e-8 to avoid division by zero 
    p = weight / (ops.sum(weight, axis=1, keepdims=True) + 1e-10)
    return np.nan_to_num(p)


def density_scale(t, select_size, indices=None):
    """
    Compute a sample of pairwise distance to find the scale.
    This function is used to make two different spaces comparable when
    calculate topological loss
    """
    # Randomly subset of points to save memory
    n_points = tf.shape(t)[0]
    sample_size = tf.minimum(n_points, select_size)

    if indices is None:
        indices = tf.random.shuffle(tf.range(n_points))[:sample_size]
    
    sample = tf.gather(t, indices)

    # Compute pairwise Euclidean distances
    r = tf.reduce_sum(sample*sample, 1, keepdims=True)
    sq_dist = r - 2*tf.matmul(sample, tf.transpose(sample)) + tf.transpose(r)
    D = tf.sqrt(tf.maximum(sq_dist, 1e-12))

    avg_dist = tf.reduce_mean(D) + 1e-8
    t_scaled = t / avg_dist
    return t_scaled, indices


