import os
import pickle
from abc import ABC, abstractmethod 

import numpy as np
import scanpy as sc
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

import keras
from keras import ops, layers, models, regularizers, initializers

from .loss import NB, ZINB
from .layers import SliceLayer, ClusteringLayer, ColwiseMultLayer, ZINBComputationLayer
from .io import write_text_matrix


MeanAct = lambda x: ops.clip(ops.exp(x), 1e-5, 1e6)
DispAct = lambda x: ops.clip(ops.softplus(x), 1e-4, 1e4)


advanced_activations = ('PReLU', 'LeakyReLU')


class Autoencoder():
    def __init__(self, input_size, output_size=None, hidden_size=(256, 64, 32, 64, 256),
                 noise_sd=0., l2_coef=0., l1_coef=0., l2_enc_coef=0., l1_enc_coef=0.,
                 ridge=0., hidden_dropout=0., input_dropout=0.,
                 batchnorm=True, activation='relu', init='glorot_uniform',
                 file_path=None, debug=False, **kwargs):

        self.input_size = input_size
        self.output_size = output_size or input_size
        self.hidden_size = hidden_size
        self.noise_sd = noise_sd
        self.l2_coef = l2_coef
        self.l1_coef = l1_coef
        self.l2_enc_coef = l2_enc_coef
        self.l1_enc_coef = l1_enc_coef
        self.ridge = ridge
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.init = init
        self.file_path = file_path
        self.debug = debug
        
        self.model = None
        self.encoder = None
        self.extra_models = {}

        if isinstance(self.hidden_dropout, list):
            assert len(self.hidden_dropout) == len(self.hidden_size)
        else:
            self.hidden_dropout = [self.hidden_dropout]*len(self.hidden_size)


    def build(self):
        self.input_layer = layers.Input(shape=(self.input_size,), name='count', sparse=True)
        self.sf_layer = layers.Input(shape=(1,), name='size_factors')
        last_hidden = self.input_layer

        if self.noise_sd > 0.0:
            last_hidden = layers.GaussianNoise(self.noise_sd, name='input_noise')(last_hidden)

        if self.input_dropout > 0.0:
            last_hidden = layers.Dropout(self.input_dropout, name='input_dropout')(last_hidden)

        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name, stage = 'latent', 'latent'
            elif i < center_idx:
                layer_name, stage = 'enc%s' % i, 'encoder'
            else:
                layer_name, stage = 'dec%s' % (i-center_idx), 'decoder'

            # Regularization mapping
            l1 = self.l1_enc_coef if (self.l1_enc_coef != 0 and stage in ('latent', 'encoder')) else self.l1_coef
            l2 = self.l2_enc_coef if (self.l2_enc_coef != 0 and stage in ('latent', 'encoder')) else self.l2_coef

            last_hidden = layers.Dense(
                hid_size, activation=None, kernel_initializer=self.init,
                kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
                name=layer_name
            )(last_hidden)

            if self.noise_sd > 0.0 and stage == 'encoder':
                last_hidden = layers.GaussianNoise(self.noise_sd, name='%s_noise' % layer_name)(last_hidden)

            if self.batchnorm:
                last_hidden = layers.BatchNormalization(center=True, scale=False)(last_hidden)

            # Add each layer with activation function except latent layer
            if stage != 'latent':

                # Modern Activation lookup
                if self.activation in ('PReLU', 'LeakyReLU'):

                    if self.activation == 'PReLU':
                        last_hidden = layers.PReLU(name='%s_act' % layer_name)(last_hidden)
                    else:
                        last_hidden = layers.LeakyReLU(name='%s_act' % layer_name)(last_hidden)
                        
                else:
                    last_hidden = layers.Activation(self.activation, name='%s_act' % layer_name)(last_hidden)

                if hid_drop > 0.0:
                    last_hidden = layers.Dropout(hid_drop, name='%s_drop' % layer_name)(last_hidden)

        self.decoder_output = last_hidden
        self.build_output()


    def build_output(self):
        mean = layers.Dense(
            self.output_size, kernel_initializer=self.init,
            kernel_regularizer=regularizers.L1L2(l1=self.l1_coef, l2=self.l2_coef),
            name='mean'
        )(self.decoder_output)
        
        output = ColwiseMultLayer(name='output')([mean, self.sf_layer])
        self.model = models.Model(inputs=[self.input_layer, self.sf_layer], outputs=output)
        self.encoder = self.get_encoder()


    def save(self):
        if self.file_path:
            os.makedirs(self.file_path, exist_ok=True)
            
            # 1. Save the Keras model weights (HDF5 or .weights.h5)
            weights_path = os.path.join(self.file_path, 'weights.weights.h5')
            self.model.save_weights(weights_path)
            
            # 2. Save the metadata/hyperparameters as a simple pickle
            # We remove the actual 'model' objects before pickling to avoid errors
            temp_model = self.model
            temp_enc = self.encoder
            self.model = None
            self.encoder = None
            
            with open(os.path.join(self.file_path, 'model_meta.pickle'), 'wb') as f:
                pickle.dump(self, f)
                
            # Restore them so the object stays usable in memory
            self.model = temp_model
            self.encoder = temp_enc


    @staticmethod
    def load_from_path(path):
        # 1. Load the pickle (metadata)
        with open(os.path.join(path, 'model_meta.pickle'), 'rb') as f:
            obj = pickle.load(f)
    
        # 2. Re-build the architecture (the model was None)
        obj.build() 
    
        # 3. Load the numerical weights
        obj.load_weights(os.path.join(path, 'weights.weights.h5'))
        return obj


    def save_weights(self, filename):
        # filename should be the path to 'weights.weights.h5'
        self.model.save_weights(filename)


    def load_weights(self, filename):
        # filename should be the path to 'weights.weights.h5'
        self.model.load_weights(filename)
        self.encoder = self.get_encoder()
        self.decoder = None  # get_decoder()


    def get_encoder(self, activation=False, dropout=False):
        if dropout:
            target = 'latent_drop'
        elif activation:
            target = 'latent_act'
        else:
            target = 'latent'
            
        return models.Model(inputs=self.model.input, 
                     outputs=self.model.get_layer(target).output,
                     name='encoder')


    def get_decoder(self, activation=False, dropout=False):
        if dropout:
            target = 'latent_drop'
        elif activation:
            target = 'latent_act'
        else:
            target = 'latent'

        latent_output = self.model.get_layer(target).output
        latent_shape = ops.shape(latent_output)[1:] 
        decoder_input = layers.Input(shape=latent_shape, name='decoder_input')
        curr = decoder_input
        found = False

        for layer in self.model.layers:
            if found:
                if isinstance(layer, layers.InputLayer) and 'size_factors' in layer.name:
                    continue
                curr = layer(curr)
            
            if layer.name == target:
                found = True
        
        return models.Model(inputs=decoder_input, outputs=curr, name='decoder')


    def predict(self, adata, mode='denoise', return_info=False, copy=False):
        assert mode in ('denoise', 'latent', 'full'), 'Unknown mode'
        adata = adata.copy() if copy else adata

        inputs = {
            'count': adata.X, 
            'size_factors': adata.obs.size_factors.values
        }

        if mode in ('latent', 'full'):
            print('Calculating low dimensional representations...')
            adata.obsm['ae'] = self.encoder.predict(inputs)        
        if mode in ('denoise', 'full'):
            print('Calculating reconstructions...')
            adata.X = self.model.predict(inputs)

        return adata if copy else None


class ZINBAutoencoder(Autoencoder):
    def build_output(self):
        # 1. Parameter branches
        pi = layers.Dense(self.output_size, activation='sigmoid', kernel_initializer=self.init,
                       kernel_regularizer=regularizers.L1L2(l1=self.l1_coef, l2=self.l2_coef),
                       name='pi')(self.decoder_output)

        disp = layers.Dense(self.output_size, activation=DispAct,
                           kernel_initializer=self.init,
                           kernel_regularizer=regularizers.L1L2(l1=self.l1_coef, l2=self.l2_coef),
                           name='dispersion')(self.decoder_output)

        mean = layers.Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                       kernel_regularizer=regularizers.L1L2(l1=self.l1_coef, l2=self.l2_coef),
                       name='mean')(self.decoder_output)

        # 2. Scale the mean by Size Factors
        scaled_mean = ColwiseMultLayer(name='scaled_output')([mean, self.sf_layer])

        # 3. Bundle for the Loss [Mean, Disp, Pi]
        output = layers.Concatenate(name='zinb_bundle')([scaled_mean, disp, pi])

        # 4. Corrected Loss Assignment
        # We use the 'call' method of your ZINBComputationLayer to bridge the KerasTensor gap
        zinb_comp = ZINBComputationLayer(ridge_lambda=self.ridge, scale_factor=1.0) # scale_factor handled by ColwiseMult
        self.loss = zinb_comp.call

        # 5. Map Extra Models for return_info=True in predict()
        self.extra_models['pi'] = models.Model(inputs=[self.input_layer, self.sf_layer], outputs=pi)
        self.extra_models['dispersion'] = models.Model(inputs=[self.input_layer, self.sf_layer], outputs=disp)
        self.extra_models['mean_norm'] = models.Model(inputs=[self.input_layer, self.sf_layer], outputs=mean)

        # 6. Build the Final Model
        self.model = models.Model(inputs=[self.input_layer, self.sf_layer], outputs=output)
        self.encoder = self.get_encoder()


    def predict(self, adata, mode='denoise', return_info=False, copy=False, colnames=None):
        adata = adata.copy() if copy else adata

        inputs = {
            'count': adata.X, 
            'size_factors': adata.obs.size_factors.values
        }

        if return_info:
            # Predict using the sub-models mapped in build_output
            adata.obsm['zinb_ae_dispersion'] = self.extra_models['dispersion'].predict(inputs, verbose=0)
            adata.obsm['zinb_ae_dropout'] = self.extra_models['pi'].predict(inputs, verbose=0)

        if mode in ('denoise', 'full'):
            print('Calculating reconstructions...')
            raw_pred = self.model.predict(inputs, verbose=0)
            # Slice only the mean (first 1/3 of the bundle)
            n_genes = raw_pred.shape[1] // 3
            adata.X = raw_pred[:, :n_genes]

        if mode in ('latent', 'full'):
            print('Calculating low dimensional representations...')
            adata.obsm['zinb_ae'] = self.encoder.predict(inputs, verbose=0)

        return adata if copy else None


class DEC(ZINBAutoencoder):
    def __init__(self, n_clusters, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha


    def build_output(self):
        super().build_output()
        self.zinb_ae = self.model
        curr = self.input_layer
        
        try:
            latent_tensor = self.zinb_ae.get_layer('latent_act').output
        except ValueError:
            latent_tensor = self.zinb_ae.get_layer('latent').output

        clustering_layer = ClusteringLayer(
            self.n_clusters, 
            alpha=self.alpha, 
            name='clustering')(latent_tensor)
        
        self.model = models.Model(inputs=self.zinb_ae.inputs,
                                  outputs=[clustering_layer, self.zinb_ae.output],
                                  name='scDEC_model')
        
        self.encoder = self.get_encoder()


    def get_encoder(self):
        """
        Deterministic encoder for DEC phase. 
        Accepts both inputs to maintain compatibility with train_step, 
        but only processes counts through a noise-free path.
        """
        hidden = self.input_layer
        
        # Extract true layers
        found_latent = False
        for layer in self.model.layers:
            # Skip noise, dropout, and any input layers
            if isinstance(layer, (layers.InputLayer, layers.GaussianNoise, layers.Dropout)):
                continue
            
            # Skip the size factor input layer
            if 'size_factors' in layer.name:
                continue

            hidden = layer(hidden)
            
            # Stop immediately after we get the bottle neck layer
            if layer.name == 'latent' or layer.name == 'latent_act':
                found_latent = True
                break

        if not found_latent:
            print("Warning: Latent layer not found during encoder extraction.")
        
        return models.Model(inputs=self.model.input, outputs=hidden, name='encoder')
    

    def get_initial_clusters(self, adata, n_neighbors=20, resolution=0.8):
        """Detects initial clusters in latent space using Leiden."""
        # 1. Get latent representation
        inputs = {
            'count': adata.X, 
            'size_factors': adata.obs.size_factors.values
        }
        z = self.encoder.predict(inputs, verbose=0)
    
        # 2. Community detection
        adata_latent = sc.AnnData(z)
        sc.pp.neighbors(adata_latent, n_neighbors=n_neighbors, use_rep="X")
        sc.tl.leiden(adata_latent, resolution=resolution, flavor='igraph', n_iterations=2)
    
        # 3. Calculate labels and centroids
        labels = adata_latent.obs['leiden'].astype(int).values
        centroids = np.array([z[labels == i].mean(axis=0) for i in sorted(np.unique(labels))])
    
        return centroids, labels
    

    def init_clustering_layer(self, n_clusters, weights):
        """Dynamically re-initializes the clustering layer with detected centroids."""
        self.n_clusters = n_clusters
        # Rebuild the model with the new number of clusters
        self.build_output() 
        # Set the centroids found by Leiden
        self.model.get_layer(name='clustering').set_weights([weights])
    

    def predict(self, adata, mode='clustering', return_info=False, copy=False):
        assert mode in ('clustering'), 'This model is used for clustering.'
        adata = adata.copy() if copy else adata

        inputs = {
            'count': adata.X, 
            'size_factors': adata.obs.size_factors.values
        }
        
        print('Calculating clustering...')
        # 1. Extract Latent space and Cluster Probabilities
        adata.obsm['stc'] = self.encoder.predict(inputs, verbose=0)
        q, denoise = self.model.predict(inputs, verbose=0)
    
        adata.obsm['stc_probs'] = q
    
        # 2. Initial Argmax (might result in skipping numbers like 5, 6)
        raw_clusters = np.argmax(q, axis=1)
    
        # 3. Standardize to 0, 1, 2... (The "Cleanup" Step)
        # This ensures [5, 6] -> [0, 1] or [0, 2, 9] -> [0, 1, 2]
        le = LabelEncoder()
        standardized_clusters = le.fit_transform(raw_clusters)
    
        # 4. Save to adata as strings/categories
        adata.obs['stc_cluster'] = standardized_clusters.astype(str)
        adata.obs['stc_cluster'] = adata.obs['stc_cluster'].astype('category')

        print(f'Identified {len(le.classes_)} unique clusters.')

        return adata if copy else None


network_options = {'ae': Autoencoder, 'zinb': ZINBAutoencoder, 'dec': DEC}



