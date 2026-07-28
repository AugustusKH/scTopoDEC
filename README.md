# scTopoDEC 

**Topological deep embedded clustering for single-cell RNA-seq data by persistent homology.**

## Background

Clustering remains a challenging task in scRNA-seq analysis despite the development of numerous computational methods. Here, we introduce **scTopoDEC (single-cell topological deep embedded clustering)**, a deep learning-based clustering method that incorporates **persistent homology** to improve clustering performance by preserving topological information in single-cell data.

<p align="center">
  <img src="figure/scTopoDEC_architecture.png" width="600" title="Analysis Workflow">
</p>

## Installation

scTopoDEC can be installed from GitHub directly via the code below:

```bash
pip install git+https://github.com/AugustusKH/scTopoDEC.git
```

## Quick start

scTopoDEC provides two execution options: a command-line interface and a Jupyter Notebook. We require an `.h5ad` file as the input in the model. As scTopoDEC is designed specifically for cell clustering, it processes the input .h5ad file, performs clustering, and writes the predicted cluster labels back to the original file as the output.

### Command line

We run the command line for scTopoDEC as follows:

```bash
scTopoDEC input_file.h5ad \
  --n_clusters 0 \
  --loss_weights (1.0, 10.0, 0.1, 1.0) \
  --output output_file.h5ad
```

If the exact number of clusters is unknown, set `--n_clusters` to 0. The model will then automatically estimate the optimal number of clusters. The remaining parameters can be configured manually as described below; however, we recommend using the default settings.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--hidden_size` | `str` | `(256, 32, 256)` | Network architecture as a tuple string |
| `--loss_weights` | `str` | `(1.0, 10.0, 0.1, 1.0)` | Loss weights `(ZINB, Cluster, SoftK, Topo)` as a tuple string |
| `--n_clusters` | `str` | `0` | Number of clusters, if set to 0, will automatically determine optimal number of clusters |
| `--n_top_genes` | `int` | `2000` | Number of highly variable genes used in model training |
| `--batch_key` | `str` | `None` | `adata.obs` key for batch labels |
| `--noise_sd` | `float` | `0.4` | Standard deviation of Gaussian noise added to input |
| `--hidden_dropout` | `float` | `0.05` | Dropout rate applied to hidden layers (0.0 to 1.0) |
| `--pretrain_epochs` | `int` | `800` | Autoencoder pre-training epochs |
| `--epochs` | `int` | `500` | Max training epochs |
| `--pretrain_lr` | `float` | `1e-3` | Pre-training learning rate |
| `--lr` | `float` | `1e-4` | Learning rate |
| `--update_interval` | `int` | `5` | Epochs between target distribution updates |
| `--tol` | `float` | `1e-3` | Convergence threshold |
| `--maximum_edge_length` | `float` | `1e-3` | Filtration cutoff when running persistent homology |
| `--topo_size` | `int` | `256` | Number of cells sampled for topological loss calculation |
| `--order` | `float` | `1.0` | Wasserstein exponent q for diagram distance |
| `--k` | `int` | `100` | Nearest neighbors for graph construction |

### Jupyter notebook

For the Jupyter Notebook interface, import the scTopoDEC package before running the analysis, as shown below:

```bash
import scTopoDEC as stc
```

After that, we can run the model based on `adata` files as follows:

```bash
stc.scTopoDEC(adata, n_clusters=0, loss_weights=(1.0, 10.0, 0.1, 1.0))
```

After the model has finished running, the scTopoDEC-derived cluster labels are stored in the same `adata` object. The clustering results can be accessed via `adata.obs['stc_cluster']`, while the latent representations learned by the model are available in `adata.obsm['X_stc']`.

We can manually set the parameters similar to the command line running as below:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hidden_size` | `tuple` or `list` | `(256, 32, 256)` | Network architecture as a tuple string |
| `loss_weights` | `tuple` or `list` | `(1.0, 10.0, 0.1, 1.0)` | Loss weights `(ZINB, Cluster, SoftK, Topo)` as a tuple string |
| `n_clusters` | `int` or `str` | `0` | Number of clusters, if set to 0, will automatically determine optimal number of clusters |
| `n_top_genes` | `int` | `2000` | Number of highly variable genes used in model training |
| `batch_key` | `str` | `None` | `adata.obs` key for batch labels |
| `noise_sd` | `float` | `0.4` | Standard deviation of Gaussian noise added to input |
| `hidden_dropout` | `float` | `0.05` | Dropout rate applied to hidden layers (0.0 to 1.0) |
| `pretrain_epochs` | `int` | `800` | Autoencoder pretraining epochs |
| `epochs` | `int` | `500` | Max training epochs |
| `pretrain_lr` | `float` | `1e-3` | Pretraining learning rate |
| `lr` | `float` | `1e-4` | Learning rate |
| `update_interval` | `int` | `5` | Epochs between target distribution updates |
| `tol` | `float` | `1e-3` | Convergence threshold |
| `maximum_edge_length` | `float` | `1e-3` | Filtration cutoff when running persistent homology |
| `topo_size` | `int` | `256` | Number of cells sampled for topological loss calculation |
| `order` | `float` | `1.0` | Wasserstein exponent q for diagram distance |
| `k` | `int` | `100` | Nearest neighbors for graph construction |
| `verbose` | `bool` | `False` | Outputs detailed training progress and model summaries |

## Running scTopoDEC on large datasets

To handle large datasets, we designed scTopoDEC to transfer the weights learned during denoising pre-training to the subsampled cells for clustering optimisation and topological training. The trained model weights are then used to predict cluster assignments for the remaining cells. 

To run scTopoDEC on large datasets, import the large-scale model module as shown below:

```bash
from scTopoDEC.api_utils import run_scTopoDEC_large_data
```

Next, we wil run the model on the large dataset. Before that, we will define standard parameters for scTopoDEC running as the dictionary data structure below:

```bash
stc_settings = {'n_clusters': 0, 'loss_weights': (1.0, 10.0, 0.1, 1.0)}
```

The model can run as follows:

```bash
large_adata, _ = run_scTopoDEC_large_data(large_adata, sampling_cells=2000, **stc_settings)
```

We can set the number of sampling cells from the option `sampling_cells`. We also can perform resapling by changing the number of sampling without denoising again as the model just save the pre-trained weights. We can run just only for the clustering and topological training as:

```bash
large_adata, _ = run_scTopoDEC_large_data(large_adata, sampling_cells=5000, 
                                          initial_pretrain_weights='global_pretrain_weights.weights.h5', 
                                          **stc_settings)
```

For the number of subsampled cells, we recommend sampling at least 10–20% of the total number of cells. All sampling parameters are listed below:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sampling_threshold` | `int` | `10000` | Sampling is done based on this threshold |
| `sampling_cells` | `int` | `2000` | Number of cells to subsample for clustering and topological training |
| `initial_pretrain_weights` | `str` or `None` | `None` | Path to pre-trained weights |
| `leiden_subsampling` | `bool` | `False` | Performs stratified subsampling using Leiden clusters to ensure diverse cell representation |
| ` leiden_resolution` | `float` | `0.5` | Resolution parameter for the Leiden-based stratified subsampling |
