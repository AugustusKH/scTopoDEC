from anndata import read_h5ad
# Import modern preprocessing functions from scanpy.pp
from scanpy.pp import (
    normalize_total, 
    highly_variable_genes, 
    log1p, 
    scale,
    filter_cells,
    filter_genes
)

# Internal module imports
from . import tools
from . import models
from . import datasets

# API Exposure
from .models.desc import train
from .tools.test import run_desc_test
from .tools.read import read_10X
from .tools.write import write_desc_result
from .tools.preprocessing import scale_bygroup

# Version control
__version__ = '2.1.1'

# Optional: Define __all__ to control 'from package import *' behavior
__all__ = [
    'read_h5ad',
    'normalize_total',
    'highly_variable_genes',
    'log1p',
    'scale',
    'filter_cells',
    'filter_genes',
    'train',
    'run_desc_test',
    'read_10X',
    'write_desc_result',
    'scale_bygroup'
]