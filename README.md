# scTopoDEC 

**Topological deep embedded clustering for single-cell RNA-seq data.**

Clustering remains a challenging task in scRNA-seq analysis despite the development of numerous computational methods. Here, we introduce **scTopoDEC (single-cell topological deep embedded clustering)**, a deep learning-based clustering method that incorporates **persistent homology** to improve clustering performance by preserving topological information in single-cell data.

<p align="center">
  <img src="figure/scTopoDEC_architecture.png" width="600" title="Analysis Workflow">
</p>

## Installation

```bash
pip install git+https://github.com/AugustusKH/scTopoDEC.git
```

## Quick start

scTopoDEC provides two execution options: a command-line interface and a Jupyter Notebook. We require an `.h5ad` file as the input in the model.

### Command line

We run the command line for scTopoDEC as follows:

```bash
scTopoDEC input_file.h5ad \
  --epochs 300 \
  --pretrain_epochs 500 \
  --n_clusters 0 \
  --output output_file.h5ad
```

If the exact number of clusters is unknown, set `--n_clusters` to 0. The model will then automatically estimate the optimal number of clusters.