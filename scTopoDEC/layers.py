import numpy as np
import keras
from keras import ops, Layer, InputSpec
import tensorflow as tf
import gudhi as gd
from gudhi import RipsComplex


class SliceLayer(keras.layers.Layer):
    """
    Slices a specific tensor from a list of input tensors.

    This layer is typically used when a previous layer or model returns 
    multiple outputs (e.g., the Mean (mu), Dispersion (theta), and 
    Dropout (pi) parameters in a ZINB model) and only one is required 
    for the next stage of the pipeline.

    # Arguments
        index: Integer, the 0-indexed position of the tensor to retrieve.

    # Input shape
        List of tensors: [(batch, d1, d2, ...), (batch, d1, d2, ...), ...]

    # Output shape
        The shape of the tensor at the specified index.
    """
    def __init__(self, index, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input should be a list')
        super().build(input_shape)

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('SliceLayer input is not a list')
        return inputs[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]

    def get_config(self):
        config = {'index': self.index}
        base_config = super().get_config()
        return {**base_config, **config}


class ClusteringLayer(keras.layers.Layer):
    """
    Clustering layer converts sample input (features or genes) to soft label, i.e. a vector that 
    represents the probability, calculated with student's t-distribution, of the sample belonging to each cluster.
    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=keras.config.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer='glorot_uniform',
            name='clusters',
            trainable=True
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            self.initial_weights = None
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        dist = ops.sum(ops.square(ops.expand_dims(inputs, axis=1) - self.clusters), axis=2)
        q = 1.0 / (1.0 + (dist / self.alpha))
        q = ops.power(q, (self.alpha + 1.0) / 2.0)
        q = q / (ops.sum(q, axis=1, keepdims=True) + 1e-10) # Add 1e-8 to avoid division by zero 
        return ops.clip(q, 1e-10, 1.0) 

    def compute_output_shape(self, input_shape):
        assert len(input_shape) == 2
        return (input_shape[0], self.n_clusters)

    def get_config(self):
        config = {'n_clusters': self.n_clusters, 'alpha': self.alpha}
        base_config = super().get_config()
        return {**base_config, **config}


class ColwiseMultLayer(keras.layers.Layer):
    """
    Multiplies each column of a matrix by a vector (Size Factors).
    Input: [Mean_Matrix (Batch, Genes), Size_Factors (Batch, 1)]
    Output: Scaled_Matrix (Batch, Genes)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs[0] is the mean, inputs[1] is the size factor
        # ops.reshape ensures the size factor is (Batch, 1) for broadcasting
        return inputs[0] * ops.reshape(inputs[1], (-1, 1))

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}


class ZINBComputationLayer(keras.layers.Layer):
    """
    Computes the ZINB loss by extracting parameters from a bundled output.
    
    Inputs: 
        y_true: The raw count matrix (Batch, Genes)
        y_pred_all: Concatenated tensor [Mean, Dispersion, Pi] (Batch, Genes * 3)
        
    Mathematical Components:
        - NB Component: log-gamma based Negative Binomial loss.
        - Zero Component: Probability of dropout vs biological zeros.
        - Ridge: L2 regularization on the dropout probability (pi).
    """
    def __init__(self, ridge_lambda=0.0, eps=1e-10, scale_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.ridge_lambda = ridge_lambda
        self.eps = eps
        self.scale_factor = scale_factor

    def call(self, y_true, y_pred_all):
        """
        Calculates the ZINB loss while handling symbolic KerasTensors.
        """
        # 1. Handle sparsity for y_true
        if isinstance(y_true, tf.SparseTensor):
            y_true_dense = tf.sparse.to_dense(y_true)
        else:
            y_true_dense = y_true

        y_true_dense = ops.cast(y_true_dense, "float32")
        n_genes = ops.shape(y_true_dense)[1]

        # 2. Extract [mean, dispersion, pi] from bundled y_pred_all
        y_pred = y_pred_all[:, :n_genes]
        theta = ops.minimum(ops.cast(y_pred_all[:, n_genes:2*n_genes], "float32"), 1e6)
        pi = y_pred_all[:, 2*n_genes:3*n_genes]

        # 3. Create mask for zeros vs non-zeros
        nonzero_mask = ops.cast(ops.greater(y_true_dense, 1e-8), "float32")

        # 4. NB component (calculated for non-zero entries)
        # Includes Log-Gamma terms for NB likelihood
        t1 = (tf.math.lgamma(theta + self.eps) + 
              tf.math.lgamma(y_true_dense + 1.0) - 
              tf.math.lgamma(y_true_dense + theta + self.eps))
        
        t2 = ((theta + y_true_dense) * ops.log(1.0 + (y_pred / (theta + self.eps))) + 
              (y_true_dense * (ops.log(theta + self.eps) - ops.log(y_pred + self.eps))))

        nb_case = (t1 + t2) - ops.log(1.0 - pi + self.eps)
        
        # 5. Zero-inflation component (calculated for zero entries)
        zero_nb = ops.power(theta / (theta + y_pred + self.eps), theta)
        zero_case = -ops.log(pi + ((1.0 - pi) * zero_nb) + self.eps)
        
        # 6. Combine cases using the mask
        result = nonzero_mask * nb_case + (1.0 - nonzero_mask) * zero_case

        # 7. Apply Ridge regularization on dropout probability (pi)
        if self.ridge_lambda > 0:
            result += self.ridge_lambda * ops.square(pi)
        
        return ops.mean(result)
    

############################
# Vietoris-Rips filtration #
############################

def _Rips(DX, max_edge, dimensions, homology_coeff_field):
    # Parameters: DX (distance matrix),
    #             max_edge (maximum edge length for Rips filtration),
    #             dimensions (homology dimensions)

    # Compute the persistence pairs with Gudhi
    rc = RipsComplex(distance_matrix=DX, max_edge_length=max_edge)
    st = rc.create_simplex_tree(max_dimension=max(dimensions) + 1)
    st.compute_persistence(homology_coeff_field=homology_coeff_field)
    
    L_indices = []
    for dimension in dimensions:

        if dimension == 0:
            finite_pairs = pairs[0]
            essential_pairs = pairs[2]
        else:
            finite_pairs = (
                pairs[1][dimension - 1]
                if len(pairs[1]) >= dimension
                else np.empty(shape=[0, 4])
            )
            essential_pairs = (
                pairs[3][dimension - 1]
                if len(pairs[3]) >= dimension
                else np.empty(shape=[0, 2])
            )

        finite_indices = np.array(finite_pairs.flatten(), dtype=np.int32)
        essential_indices = np.array(essential_pairs.flatten(), dtype=np.int32)

        L_indices.append((finite_indices, essential_indices))

    return L_indices


class RipsLayer(tf.keras.layers.Layer):
    """
    Modified TensorFlow layer for computing Rips persistence.
    Supports both point clouds and pre-computed distance matrices.
    """
    def __init__(self, homology_dimensions, maximum_edge_length=np.inf, 
                 min_persistence=None, homology_coeff_field=11, **kwargs):
        """
        Constructor for the RipsLayer class

        Parameters:
            maximum_edge_length (float): maximum edge length for the Rips complex
            homology_dimensions (List[int]): list of homology dimensions
            min_persistence (List[float]): minimum distance-to-diagonal of the points 
                in the output persistence diagrams (default None, in which case 0. is 
                used for all dimensions)
            homology_coeff_field (int): homology field coefficient. Must be a prime number. 
                Default value is 11. Max is 46337.
        """
        super().__init__(**kwargs)
        self.max_edge = maximum_edge_length
        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence if min_persistence is not None else [0.0 for _ in range(len(self.dimensions))]
        self.hcf = homology_coeff_field

    def call(self, X, is_distance_matrix=False):
        """
        Compute Rips persistence diagram associated to a point cloud

        Parameters:
            X: Point cloud [N, D] OR Distance Matrix [N, N]
            is_distance_matrix: Boolean flag

        Returns:
            List[Tuple[tf.Tensor,tf.Tensor]]: List of Rips persistence diagrams. 
            The length of this list is the same than that of dimensions, i.e., 
            there is one persistence diagram per homology dimension provided in 
            the input list dimensions. Moreover, the finite and essential parts of 
            the persistence diagrams are provided separately: each element of this 
            list is a tuple of size two that contains the finite and essential parts of 
            the corresponding persistence diagram, of shapes [num_finite_points, 2] and 
            [num_essential_points, 1] respectively
        """
        # Compute distance matrix
        if is_distance_matrix:
            # X is already the distance matrix DX
            DX = X
        else:
            # Compute distance matrix from coordinates
            DX = tf.norm(tf.expand_dims(X, 1) - tf.expand_dims(X, 0), axis=2)

        # The rest of the GUDHI internal logic remains the same
        indices = _Rips(DX.numpy(), self.max_edge, self.dimensions, self.hcf)
        
        # Get persistence diagrams by simply picking the corresponding entries in the distance matrix
        self.dgms = []
        for idx_dim, dimension in enumerate(self.dimensions):
            cur_idx = indices[idx_dim]
            if dimension > 0:
                finite_dgm = tf.reshape(tf.gather_nd(DX, tf.reshape(cur_idx[0], [-1, 2])), [-1, 2])
                essential_dgm = tf.reshape(tf.gather_nd(DX, tf.reshape(cur_idx[1], [-1, 2])), [-1, 1])
            else:
                reshaped_cur_idx = tf.reshape(cur_idx[0], [-1, 3])
                finite_dgm = tf.concat([
                    tf.zeros([reshaped_cur_idx.shape[0], 1]),
                    tf.reshape(tf.gather_nd(DX, reshaped_cur_idx[:, 1:]), [-1, 1])
                ], axis=1)
                essential_dgm = tf.zeros([cur_idx[1].shape[0], 1])
            
            # Apply min_persistence filtering...
            min_pers = self.min_persistence[idx_dim]
            persistent_indices = tf.where(tf.math.abs(finite_dgm[:, 1] - finite_dgm[:, 0]) > min_pers)
            self.dgms.append((tf.reshape(tf.gather(finite_dgm, indices=persistent_indices), [-1, 2]), essential_dgm))
            
        return self.dgms