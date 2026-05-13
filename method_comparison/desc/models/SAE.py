import os
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping



class SAE(object):
    """ 
    Stacked autoencoders. It can be trained in layer-wise manner followed by end-to-end fine-tuning.
    For a 5-layer (including input layer) example:
        Autoendoers model: Input -> encoder_0->act -> encoder_1 -> decoder_1->act -> decoder_0;
        stack_0 model: Input->dropout -> encoder_0->act->dropout -> decoder_0;
        stack_1 model: encoder_0->act->dropout -> encoder_1->dropout -> decoder_1->act;
    
    Usage:
        from SAE import SAE
        sae = SAE(dims=[784, 500, 10])  # define a SAE with 5 layers
        sae.fit(x, epochs=100)
        features = sae.extract_feature(x)
        
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
              The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation (default='relu'), not applied to Input, Hidden and Output layers.
        drop_rate: drop ratio of Dropout for constructing denoising autoencoder 'stack_i' during layer-wise pretraining
        batch_size: `int`, optional. Default:`256`, the batch size for autoencoder model and clustering model.
        random_seed, `int`,optional. Default,`201809`. the random seed for random.seed,,,numpy.random.seed,tensorflow.set_random_seed
        actincenter: the activation function in last layer for encoder and last layer for encoder (avoiding the representation values and reconstruct outputs are all non-negative)
        init: `str`,optional. Default: `glorot_uniform`. Initialization method used to initialize weights.
        use_earlyStop: optional. Default,`True`. Stops training if loss does not improve if given min_delta=1e-4, patience=10.
        save_dir:'str',optional. Default,'result_tmp',some result will be saved in this directory.
    """
    def __init__(self, dims, act='relu', 
                 drop_rate=0.2, 
                 batch_size=32,
                 random_seed=201809,
                 actincenter="tanh",
                 init="glorot_uniform",
                 use_earlyStop=True,
                 save_dir='result_tmp'):
        
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.activation = act
        self.actincenter = actincenter
        self.drop_rate = drop_rate
        self.init = init
        self.batch_size = batch_size
        self.use_earlyStop = use_earlyStop
        
        # Modern Random Seeding
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        self.random_seed = random_seed
        # Initialize the stack of denoising autoencoders
        self.stacks = [self.make_stack(i, seed=self.random_seed + i) for i in range(self.n_stacks)]
        # Build the global autoencoder and the encoder-only model
        self.autoencoders, self.encoder = self.make_autoencoders()


    def get_init(self, seed=None):
        """Returns a modern Keras initializer object."""
        inits = {
            'glorot_uniform': keras.initializers.GlorotUniform(seed=seed),
            'glorot_normal': keras.initializers.GlorotNormal(seed=seed),
            'he_normal': keras.initializers.HeNormal(seed=seed),
            'he_uniform': keras.initializers.HeUniform(seed=seed),
            'truncated_normal': keras.initializers.TruncatedNormal(stddev=0.05, seed=seed),
            'random_normal': keras.initializers.RandomNormal(stddev=0.04, seed=seed)
        }
        return inits.get(self.init.lower(), inits['glorot_uniform'])
        
        
    def make_autoencoders(self):
        """Builds the full symmetric autoencoder."""
        x = Input(shape=(self.dims[0],), name='input')
        h = x

        # Encoder
        for i in range(self.n_stacks - 1):
            h = Dense(self.dims[i + 1], 
                      kernel_initializer=self.get_init(seed=self.random_seed + i),
                      activation=self.activation, 
                      name=f'encoder_{i}')(h)

        # Bottleneck (Latent Layer)
        h = Dense(self.dims[-1], 
                  kernel_initializer=self.get_init(seed=self.random_seed + self.n_stacks),
                  name=f'encoder_{self.n_stacks - 1}', 
                  activation=self.actincenter)(h)

        y = h
        # Decoder
        for i in range(self.n_stacks - 1, 0, -1):
            y = Dense(self.dims[i], 
                      kernel_initializer=self.get_init(seed=self.random_seed + self.n_stacks + i),
                      activation=self.activation, 
                      name=f'decoder_{i}')(y)

        # Output Reconstruction
        y = Dense(self.dims[0], 
                  kernel_initializer=self.get_init(seed=self.random_seed + 2 * self.n_stacks),
                  name='decoder_0', 
                  activation=self.actincenter)(y)

        return Model(inputs=x, outputs=y, name="AE"), Model(inputs=x, outputs=h, name="encoder")


    def make_stack(self, ith, seed=0):
        """ 
        Make the ith denoising autoencoder for layer-wise pretraining. It has single hidden layer. The input data is 
        corrupted by Dropout(drop_rate)
        
        Arguments:
            ith: int, in [0, self.n_stacks)
        """
        in_out_dim = self.dims[ith]
        hidden_dim = self.dims[ith+1]
        
        # Logic for middle vs output layer activations
        output_act = self.actincenter if ith == 0 else self.activation
        hidden_act = self.actincenter if ith == self.n_stacks - 1 else self.activation

        model = Sequential([
            Dropout(self.drop_rate, input_shape=(in_out_dim,), seed=seed),
            Dense(units=hidden_dim, activation=hidden_act, 
                  kernel_initializer=self.get_init(seed=seed), name=f'encoder_{ith}'),
            Dropout(self.drop_rate, seed=seed + 1),
            Dense(units=in_out_dim, activation=output_act, 
                  kernel_initializer=self.get_init(seed=seed + 1), name=f'decoder_{ith}')
        ])
        return model


    def pretrain_stacks(self, x, epochs=200, decaying_step=3):
        """ 
        Layer-wise pretraining. Each stack is trained for 'epochs' epochs using SGD with learning rate decaying 10
        times every 'epochs/3' epochs.
        
        Arguments:
            x: input data, shape=(n_samples, n_dims)
            epochs: epochs for each stack
            decayiing_step: learning rate multiplies 0.1 every 'epochs/decaying_step' epochs 
        """
        features = x

        for i in range(self.n_stacks):
            print(f'--- Pretraining Layer {i+1}/{self.n_stacks} ---')
            for j in range(int(decaying_step)):
                lr = pow(10, -1 - j)
                print(f'Learning rate: {lr}')
                self.stacks[i].compile(optimizer=SGD(learning_rate=lr, momentum=0.9), loss='mse')
                
                callbacks = []
                if self.use_earlyStop:
                    callbacks.append(EarlyStopping(monitor='loss', min_delta=1e-4, patience=10))
                
                self.stacks[i].fit(features, features, 
                                   batch_size=self.batch_size, 
                                   epochs=math.ceil(epochs / decaying_step),
                                   callbacks=callbacks, verbose=1)

            # Pass features through the current encoder to get inputs for the next stack
            # We use the internal layers directly to avoid the 'sequential has no defined input' error
            current_stack = self.stacks[i]
            
            # Create a functional model using the layers of the stack
            # This is more robust in Keras 3 for extracting intermediate features
            input_layer = current_stack.layers[0] # The Dropout layer
            encoder_layer = current_stack.get_layer(f'encoder_{i}')
            
            # Construct a temporary model to extract the bottleneck features
            feature_model = Model(inputs=current_stack.input, outputs=encoder_layer.output)
            features = feature_model.predict(features)


    def pretrain_autoencoders(self, x, epochs=300):
        """
        Fine tune autoendoers end-to-end after layer-wise pretraining using 'pretrain_stacks()'
        Use SGD with learning rate = 0.1, decayed 10 times every 80 epochs
        
        Arguments:
        x: input data, shape=(n_samples, n_dims)
        epochs: training epochs
        """
        print('Transferring pretrained weights to global AE...')

        for i in range(self.n_stacks):
            name = f'encoder_{i}'
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())
            
            # Decoders are trained greedily too, we sync them here
            name_dec = f'decoder_{i}'
            self.autoencoders.get_layer(name_dec).set_weights(self.stacks[i].get_layer(name_dec).get_weights())

        print('Fine-tuning end-to-end...')

        for j in range(math.ceil(epochs / 50)):
            lr = pow(10, -j)
            print(f'Learning rate: {lr}')
            self.autoencoders.compile(optimizer=SGD(learning_rate=lr, momentum=0.9), loss='mse')
            
            callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=10)]
            self.autoencoders.fit(x, x, batch_size=self.batch_size, epochs=50, callbacks=callbacks)


    def fit(self, x, epochs=300, decaying_step=3): # use stacked autoencoder pretrain and fine tuning
        self.pretrain_stacks(x, epochs=int(epochs/2),decaying_step=decaying_step)
        self.pretrain_autoencoders(x, epochs=epochs)


    def extract_feature(self, x):
        """
        Extract features from the middle layer of autoencoders(representation).
        
        Arguments:
        x: data
        """
        return self.encoder.predict(x)


if __name__ == "__main__":
    """
    Standard verification test using MNIST.
    Run: python SAE.py
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi
    import os

    def load_mnist(sample_size=10000):
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Merge and flatten
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = x.reshape((x.shape[0], -1)).astype('float32')
        
        # CRITICAL: Scale to [0, 1] for stable SAE training
        x /= 255.0
        
        print(f'MNIST combined shape: {x.shape}')
        
        # Random subsample
        indices = np.random.choice(x.shape[0], sample_size, replace=False)
        return x[indices], y[indices]

    # Force CPU for the test to avoid Colab/Local GPU memory conflicts
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # 1. Load Data
    x_test_data, y_true = load_mnist(10000)
    db_name = 'mnist_test'
    n_clusters = 10

    # 2. Define and train SAE model
    # dims: Input(784) -> Hidden(64) -> Latent(32)
    sae = SAE(dims=[x_test_data.shape[-1], 64, 32], save_dir='mnist_results')
    
    print("Step 1: Starting SAE training (Pretraining + Fine-tuning)...")
    sae.fit(x=x_test_data, epochs=100) # Reduced epochs for a quick test
    
    # Save weights for later use
    weights_path = f'weights_{db_name}.weights.h5' # Modern Keras 3 extension
    sae.autoencoders.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")

    # 3. Evaluation
    print('Step 2: Extracting features...')
    features = sae.extract_feature(x_test_data)
    
    print('Step 3: Performing K-Means clustering on latent space...')
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    y_pred = km.fit_predict(features)
    
    score = nmi(y_true, y_pred)
    print(f'\n--- TEST RESULT ---')
    print(f'K-means NMI on SAE Latent Space: {score:.4f}')
    print(f'-------------------')
