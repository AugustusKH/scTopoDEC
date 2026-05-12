import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda
import tensorflow.keras.backend as K

class ConstantDispersionLayer(Layer):
    '''
    An identity layer which allows us to inject extra parameters
    such as dispersion to Keras models
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.theta = self.add_weight(shape=(1, input_shape[1]),
                                     initializer='zeros',
                                     trainable=True,
                                     name='theta')
        self.theta_exp = tf.clip_by_value(K.exp(self.theta), 1e-3, 1e4)
        super().build(input_shape)

    def call(self, x):
        return tf.identity(x)

    def compute_output_shape(self, input_shape):
        return input_shape


class SliceLayer(Layer):
    def __init__(self, index, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input should be a list')
        super().build(input_shape)

    def call(self, x):
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]


# Replaced tf.is_nan with modern tf.math.is_nan
nan2zeroLayer = Lambda(lambda x: tf.where(tf.math.is_nan(x), tf.zeros_like(x), x))

# Optimized using modern TF2 dynamic broadcasting instead of manual tf.ones matrix multiplication
ColWiseMultLayer = lambda name: Lambda(
    lambda l: l[0] * tf.cast(tf.reshape(l[1], (-1, 1)), l[0].dtype), 
    name=name
)