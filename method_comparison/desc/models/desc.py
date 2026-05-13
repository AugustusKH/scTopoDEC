# Copyright 2018 Xiangjie Li,Yafei Lyu,Mingyao Li, Gang Hu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import math
import random
import multiprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from time import time as get_time

try:
    from .network import *
except:
    from network import *
    
#or 
def getdims(shape):
    """
    This function will give the suggested nodes for each encoder layer
    return the dims for network
    """
    n_sample = shape[0]
    n_features = shape[1]
    
    if n_sample > 20000:
        return [n_features, 128, 32]
    elif n_sample > 10000:
        return [n_features, 64, 32]
    elif n_sample > 5000:
        return [n_features, 32, 16]
    elif n_sample > 2000:
        return [n_features, 128]
    else:
        return [n_features, 64]

    
def train_single(data, dims=None, alpha=1.0, tol=0.005, init='glorot_uniform',
                 louvain_resolution=1.0, n_neighbors=15, pretrain_epochs=300,
                 batch_size=256, activation='relu', actincenter='tanh',
                 drop_rate_SAE=0.2, is_stacked=True, use_earlyStop=True,
                 use_ae_weights=False, save_encoder_weights=False,
                 save_encoder_step=4, save_dir='result_tmp', max_iter=1000,
                 epochs_fit=5, num_Cores=None, use_GPU=True, GPU_id=None,
                 random_seed=201809, verbose=True, do_tsne=False,
                 do_umap=False, kernel_clustering="t"):
    
    # 1. Handle AnnData Conversion
    if isinstance(data, AnnData):
        adata = data
    else:
        adata = sc.AnnData(data)
    
    if dims is None:
        dims = getdims(adata.shape)
    
    # 2. Hardware and Seed Configuration
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    total_cpu = multiprocessing.cpu_count()
    if num_Cores is None:
        num_Cores = max(1, int(total_cpu / 2))
    
    if use_GPU and tf.config.list_physical_devices('GPU'):
        if GPU_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_id)
        print(f"Using GPU: {GPU_id if GPU_id else 'Default'}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # Optimize CPU threads for TensorFlow 2.x
        tf.config.threading.set_intra_op_parallelism_threads(num_Cores)
        tf.config.threading.set_inter_op_parallelism_threads(num_Cores)
        print(f"Using CPU with {num_Cores} cores")

    # 3. Model Initialization
    tic = get_time()
    desc = DescModel(
        dims=dims, x=adata.X, alpha=alpha, tol=tol, init=init,
        louvain_resolution=louvain_resolution, n_neighbors=n_neighbors,
        pretrain_epochs=pretrain_epochs, epochs_fit=epochs_fit,
        batch_size=batch_size, random_seed=random_seed,
        activation=activation, actincenter=actincenter,
        drop_rate_SAE=drop_rate_SAE, is_stacked=is_stacked,
        use_earlyStop=use_earlyStop, use_ae_weights=use_ae_weights,
        save_dir=save_dir
    )
    
    # KLD is the standard loss for Deep Embedding Clustering
    desc.compile(optimizer=tf.keras.optimizers.SGD(0.01, 0.9), loss='kld')
    
    # 4. Training
    Embeded_z, q_pred = desc.fit(maxiter=max_iter)
    
    # 5. Metadata Update
    res_key = f"{louvain_resolution}"
    y_pred = np.argmax(q_pred, axis=1)
    adata.obs[f'desc_{res_key}'] = pd.Categorical(y_pred.astype(str))
    adata.obsm[f'X_Embeded_z{res_key}'] = Embeded_z
    
    # 6. Optional Visualization
    if do_umap:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=f"X_Embeded_z{res_key}")
        sc.tl.umap(adata)
        adata.obsm[f"X_umap{res_key}"] = adata.obsm["X_umap"].copy()

    return adata


def train(data, louvain_resolution=[0.6, 0.8], **kwargs):
    """ Deep Embeded single cell clustering(DESC) API
    Conduct clustering for single cell data given in the anndata object or np.ndarray,sp.sparmatrix,or pandas.DataFrame
      
    
    Argument:
    ------------------------------------------------------------------
    data: :class:`~anndata.AnnData`, `np.ndarray`, `sp.spmatrix`,`pandas.DataFrame`
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    dims: `list`, the number of node in encoder layer, which include input dim, that is
    [1000,64,32] represents for the input dimension is 1000, the first hidden layer have 64 node, and second hidden layer(or bottle neck layers) have 16 nodes. if not specified, it will be decided automatically according to the sample size.
    
    alpha: `float`, optional. Default: `1.0`, the degree of t-distribution.
    tol: `float`, optional. Default: `0.005`, Stop criterion, clustering procedure will be stoped when the difference ratio betwen the current iteration and last iteration larger than tol.
    init: `str`,optional. Default: `glorot_uniform`.
        Initialization method used to initialize weights.

    louvain_resolution: `list  or str or float. for example, louvain_resolution=1.2 or louvain_resolution=[0.2,0.4,0.8] or louvain_resolution="0.3,0.4,0.8" sep with ","
    n_neighbors, `int`, optional. Default: 10. The size of local neighborhood (in terms of number of neighboring data points) used for connectivity matrix. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved. In general values should be in the range 2 to 100. Lo 

    pretrain_epochs:'int',optional. Default:`300`,the number of epochs for autoencoder model. 

    batch_size: `int`, optional. Default:`256`, the batch size for autoencoder model and clustering model. 

    activation; `str`, optional. Default,`relu`. the activation function for autoencoder model,which can be 'elu,selu,softplus,tanh,siogmid et al.', for detail please refer to`https://keras.io/activations/`.

    actincenter: `str`, optional. Default,'tanh', the activation function for the last layer in encoder and decoder model.

    drop_rate_SAE: `float`, optional. Default, `0.2`. The drop rate for Stacked autoencoder, which just for  finetuning. 

    is_stacked:`bool`,optional. Default,`True`.The model wiil be pretrained by stacked autoencoder if is_stacked==True.

    use_earlyStop:`bool`,optional. Default,`True`. Stops training if loss does not improve if given min_delta=1e-4, patience=10.

    use_ae_weights: `bool`, optional. Default, `True`. Whether use ae_weights that has been pretrained(which must saved in `save_dir/ae_weights.h5`)

    save_encoder_weights: `bool`, optional. Default, `False`, it will save inter_ae_weights for every 20 iterations. )

    save_dir: 'str',optional. Default,'result_tmp',some result will be saved in this directory.

    max_iter: `int`, optional. Default,`1000`. The maximum iteration for clustering.

    epochs_fit: `int or fload`,optional. Default,`4`, updateing clustering probability for each epochs_fit*n_sample, where n_sample is the sample size 

    num_Cores: `int`, optional. Default,`20`. How many cpus use during tranning. if `num_Cores` > the max cpus in our computer, num_Cores will use  a half of cpus in your computer. 

    use_GPU=False, `bool`, optional. Default, `True`. it will use GPU to train model if GPU is avaliable 

    GPU_id=None, `str or int`, optional.The GPU id in your device.  Default, `None`. it will use GPU to train model if `use_GPU`==True and GPU_id is not None; 

    random_seed, `int`,optional. Default,`201809`. the random seed for random.seed, numpy.random.seed, tensorflow.set_random_seed

    verbose,`bool`, optional. Default, `True`. It will ouput the model summary if verbose==True.

    do_tsne,`bool`,optional. Default, `False`. Whethter do tsne for representation or not.

    learning_rate,`float`,optional, Default(150).Note that the R-package "Rtsne" uses a default of 200. The learning rate can be a critical parameter. It should be between 100 and 1000. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high. If the cost function gets stuck in a bad local minimum increasing the learning rate helps sometimes.

    perplexity, `float`, optional, Default(30). The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. The choice is not extremely critical since t-SNE is quite insensitive to this parameter.
    do_umap, `bool`, optional. Default, `False`,Whethter do umap for representation or not
    ------------------------------------------------------------------
    """

    if isinstance(louvain_resolution, (float, int)):
        louvain_resolution = [float(louvain_resolution)]
    elif isinstance(louvain_resolution, str):
        louvain_resolution = [float(x) for x in louvain_resolution.split(',')]

    adata = data
    for res in louvain_resolution:
        print(f"\n>>> Processing Resolution: {res}")
        adata = train_single(data=adata, louvain_resolution=res, **kwargs)
    
    return adata


if __name__ == '__main__':
    # Test pipeline with modern Scanpy commands
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='pbmc.h5ad')
    args = parser.parse_args()

    # Modern Scanpy Preprocessing
    adata = sc.read(args.input)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Corrected mitochondrial calculation for modern Scanpy
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
    
    # Modernized normalization (replaced normalize_per_cell)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)

    # Run DESC
    adata = train(adata, louvain_resolution=[0.6], use_GPU=False)
    adata.write("desc_result.h5ad")

         
    
 
