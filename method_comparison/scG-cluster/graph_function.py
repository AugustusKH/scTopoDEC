import os
import argparse
import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import h5py


# topk_global = 30
# topk_local = 5

def construct_graph(features_in, label, method, topk, fname):
    num = len(label)
    print(f"Number of samples: {num}")

    # FIXED: Make a strict local copy so we do not permanently corrupt the global dataset
    features = features_in.copy()

    dist = None
    if method == 'heat':
        dist = -0.5 * pair(features, metric='manhattan') ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        dist = np.dot(features, features.T) / (np.linalg.norm(y) * np.linalg.norm(features))

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], - (topk + 1))[-(topk + 1):]
        inds.append(ind)

    counter = 0
    A = np.zeros_like(dist)
    
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    # Use 'with open' to safely handle file writing
    with open(fname, 'w') as f:
        for i, v in enumerate(inds):
            for vv in v:
                if vv == i:
                    continue
                # If labels are provided, calculate edge error rate
                if len(label) > 0 and label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
                A[i, vv] = 1
                
    print('error rate: {:.4f}'.format(counter / (num * topk)))
    return A

# ==============================================================================
# Execution Block
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct KNN graphs for scG-cluster')
    
    # Define terminal arguments
    parser.add_argument('--input_file', type=str, required=True, help='Path to the processed .h5 file')
    parser.add_argument('--output_global', type=str, required=True, help='Path to save the global graph (.txt)')
    parser.add_argument('--output_local', type=str, required=True, help='Path to save the local graph (.txt)')
    parser.add_argument('--method', type=str, default='ncos', choices=['heat', 'cos', 'ncos', 'p'], help='Similarity method')
    parser.add_argument('--topk_global', type=int, default=30, help='Number of global neighbors')
    parser.add_argument('--topk_local', type=int, default=5, help='Number of local neighbors')

    args = parser.parse_args()

    # Load Data
    with h5py.File(args.input_file, 'r') as file:
        x = np.array(file['X'])
        y = np.array(file['Y']).flatten()  
        print(f"Reading from: {args.input_file}")
        print(f"Data shape: {x.shape}")

    # Constructing an adjacency matrix of global features
    print("\n--- Constructing Global Graph ---")
    A_global = construct_graph(x, y, args.method, args.topk_global, args.output_global)

    # Constructing an adjacency matrix of local features
    print("\n--- Constructing Local Graph ---")
    A_local = construct_graph(x, y, args.method, args.topk_local, args.output_local)
    
    print("\n--- Graph Construction Complete ---")


