# scTopoDEC 

**Topological deep embedded clustering for single-cell RNA-seq data.**

Clustering remains a challenging task in scRNA-seq analysis despite the development of numerous computational methods. Here, we introduce scTopoDEC (single-cell topological deep embedded clustering), a deep learning-based clustering method that incorporates persistent homology to improve clustering performance by preserving topological information in single-cell data.

<p align="center">
  <img src="figure/scTopoDEC_architecture.png" width="600" title="Analysis Workflow">
</p>

## Quick Start

### Installation
```bash
git clone [https://github.com/AugustusKH/scTopoDEC.git](https://github.com/AugustusKH/scTopoDEC.git)
cd scTopoDEC
pip install -e .