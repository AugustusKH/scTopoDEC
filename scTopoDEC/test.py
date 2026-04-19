import random
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import gudhi as gd

from .utils import density_scale, get_topo_representation


def ph_input_test(batch_size=64, n_components=30, k=15, t=8, max_edge_length=2):
  """
  This function is used to calculate filtration and show 
  persistent diagrams (PGs) for loop (H1) based on raw adata input 
  with several representation. We show only four modes: UMAP, kNN,
  eff_res, and diffusion

  # Arguments 
  batch_size : `int`, optional (default: 32)
    Number of samples per gradient update.
  n_components : `int`, optional (default: 30)
    The number of dimensions to retain when using 'pca', 'umap', or their corresponding distance 
    modes for the input representation.
  k : `int`, optional (default: 15)
    The number of nearest neighbors used to construct the adjacency matrix for graph-based modes 
    (e.g., 'knn', 'eff_res', and 'diffusion'). 
  t : `int`, optional (default: 8)
    The diffusion time (number of power iterations) applied to the transition matrix when calculating 
    diffusion distances.
  maximum_edge_length : `float`, optional (default: 2.)
    The filtration cutoff. Limits the distance at which points are connected. 
    Prevents OOM errors by ignoring very long-distance edges.

  # Return
    Calculates and displays a 2x2 grid of Persistence Diagrams for different 
    representation modes.
  """
  # Load the example single-cell dataset
  adata = sc.datasets.paul15()
  adata_orig = adata.copy()

  # Set global seeds for reproducibility
  random.seed(0)
  np.random.seed(0)
  tf.random.set_seed(0)
  keras.utils.set_random_seed(0)

  represent_modes = ('umap', 'knn', 'eff_res', 'diffusion')
    
  # Initialize the Matplotlib figure
  fig, axes = plt.subplots(2, 2, figsize=(12, 10))
  axes = axes.flatten() 

  for i, mode in enumerate(represent_modes):
    print(f"Representation with: {mode}")
  
    # Get representation via utility wrapper
    topo_input = get_topo_representation(adata, input_mode=mode,
                                         n_components=n_components,
                                         k=k, t=t)

    # Sample the representation
    n_cells = topo_input.shape[0]
    indices = np.random.choice(n_cells, size=batch_size, replace=False)
        
    # Handle distance matrices vs coordinates for sampling
    if topo_input.ndim == 2 and topo_input.shape[0] == topo_input.shape[1]:
      topo_sample = topo_input[np.ix_(indices, indices)]
    else:
      topo_sample = topo_input[indices, :]

    # Data scaling to ensure unit average distance
    topo_sample_scale, _ = density_scale(tf.cast(topo_sample, tf.float32), batch_size)
    topo_sample_scale = topo_sample_scale.numpy() 

    # Show min-max distance before and after scaling
    dist_matrix = pairwise_distances(topo_sample)
    print("Before scaling")
    print(f"Min dist: {dist_matrix[dist_matrix > 0].min()}")
    print(f"Max dist: {dist_matrix.max()}")   

    dist_matrix = pairwise_distances(topo_sample_scale)
    print("\nAfter scaling")
    print(f"Min dist: {dist_matrix[dist_matrix > 0].min()}")
    print(f"Max dist: {dist_matrix.max()}\n")   

    # Calculate persistence diagram with GUDHI
    if mode in ['knn', 'eff_res', 'diffusion']:
      rips = gd.RipsComplex(distance_matrix=topo_sample_scale, max_edge_length=max_edge_length)
    else:
      rips = gd.RipsComplex(points=topo_sample_scale, max_edge_length=max_edge_length)
            
    st = rips.create_simplex_tree(max_dimension=2)
    dgm = st.persistence()

    # Plot to the specific subplot index
    gd.plot_persistence_diagram(dgm, axes=axes[i])
    axes[i].set_title(f"Mode: {mode.upper()}")

  plt.tight_layout()
  plt.show()
  
  return