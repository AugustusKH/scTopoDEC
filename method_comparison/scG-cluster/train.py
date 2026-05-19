import random
import argparse
from time import time
import numpy as np
import h5py
import tensorflow as tf
from scipy import sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # FIXED: Added missing PCA import

# Remove warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from model import scG
from evaluation import eva2, cluster_acc  # FIXED: Imported the safe, updated cluster_acc
from utils import *

def get_args(dataset_path, model_pth, seed=0, pretrain_epochs=800, pretrain_alpha=0.1,
             maxiter=300, train_alpha=0.1, n_pairs=0.1):

    parser = argparse.ArgumentParser(description='Parser for scG')
    parser.add_argument("--seed", default=seed, type=int)
    parser.add_argument('--dataset_path', default=dataset_path, type=str, help='path to dataset (adata)')
    
    # Pretrain
    parser.add_argument("--pretrain_epochs", default=pretrain_epochs, type=int)
    parser.add_argument("--pretrain_alpha", default=pretrain_alpha, type=float)
    parser.add_argument("--model_pth", default=model_pth, type=str)
    
    # Train
    parser.add_argument("--maxiter", default=maxiter, type=int)
    parser.add_argument("--train_alpha", default=train_alpha, type=float)
    parser.add_argument("--n_pairs", default=n_pairs, type=float)

    args = parser.parse_args()
    return args


def computeCentroids(X, labels):
    num_clusters = np.max(labels) + 1
    centroids = np.zeros((num_clusters, X.shape[1]))
    for k in range(num_clusters):
        centroids[k, :] = np.mean(X[labels == k, :], axis=0)
    return centroids


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def degree_power(A, k):   
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


if __name__ == "__main__":
    # Add the missing graph paths to your argument parser
    parser = argparse.ArgumentParser(description='Parser for scG Training')
    parser.add_argument("--seed", default=0, type=int)
    
    # Dataset and Output paths
    parser.add_argument('--dataset_path', required=True, type=str, help='Path to processed .h5 file')
    parser.add_argument('--global_graph', required=True, type=str, help='Path to global graph .txt')
    parser.add_argument('--local_graph', required=True, type=str, help='Path to local graph .txt')
    parser.add_argument('--model_pth', required=True, type=str, help='Folder to save model weights')
    
    # Hyperparameters
    parser.add_argument("--pretrain_epochs", default=800, type=int)
    parser.add_argument("--maxiter", default=300, type=int)
    parser.add_argument("--pretrain_alpha", default=0.1, type=float)
    parser.add_argument("--train_alpha", default=0.1, type=float)
    parser.add_argument("--n_pairs", default=0.1, type=float)

    args = parser.parse_args()
    print(args)

    # Seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    # Load Data
    print(f"Loading data from {args.dataset_path}...")
    with h5py.File(args.dataset_path, 'r') as file:
        X = file['X'][:]  
        y = file['Y'][:]  
        size_factor = file['size_factors'][:]
        raw = file['raw'][:]

    cluster_number = int(max(y) - min(y) + 1)
    print("Cluster number:", cluster_number)

    count = X
    raw_count = raw
    num_nodes = count.shape[0]

    # Load Global Adjacency Graph
    print(f"Loading Global Graph from {args.global_graph}...")
    edges_global = np.loadtxt(args.global_graph, dtype=int)
    G_adj = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in edges_global:
        G_adj[edge[0], edge[1]] = 1
        G_adj[edge[1], edge[0]] = 1
    np.fill_diagonal(G_adj, 1)
    G_adj_n = norm_adj(G_adj)

    # Load Local Adjacency Graph
    print(f"Loading Local Graph from {args.local_graph}...")
    edges_local = np.loadtxt(args.local_graph, dtype=int)
    L_adj = np.zeros((num_nodes, num_nodes), dtype=int)
    for edge in edges_local:
        L_adj[edge[0], edge[1]] = 1
        L_adj[edge[1], edge[0]] = 1
    np.fill_diagonal(L_adj, 1)
    L_adj_n = norm_adj(L_adj)

    # Instantiate Model
    import os
    os.makedirs(args.model_pth, exist_ok=True)
    
    model = scG(raw_count, count, size_factor, args.model_pth, G_adj, G_adj_n, L_adj, L_adj_n, n_clusters=cluster_number)

    # Pre-training
    print("--- Starting Pre-training ---")
    t0 = time()
    model.pre_train(epochs=args.pretrain_epochs)
    t1 = time()
    print(f'Pretrain run time: {t1 - t0:.2f} seconds')

    # Centroid Initialization
    print("--- Initializing Centroids ---")
    model.load_model("pretrain")
    X_pretrain = model.embedding(count)
    pca = PCA(n_components=15)
    countp = pca.fit_transform(count)
    labels = KMeans(n_clusters=cluster_number).fit(countp).labels_
    centers = computeCentroids(X_pretrain, labels)

    # Deep Clustering Training
    print("--- Starting Deep Clustering ---")
    t2 = time()
    model.train(y, epochs=args.maxiter, centers=centers)
    t3 = time()
    print(f'Train run time: {t3 - t2:.2f} seconds')
    print(f'Total pipeline run time: {t3 - t0:.2f} seconds')

    # Evaluating clustering results
    if len(y) > 0 and np.max(y) > 0:
        model.load_model("train")
        X_train, y_pred = model.get_cluster()

        acc, _ = cluster_acc(y, y_pred)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        print(f'\nFINAL RESULTS -> ACC: {acc:.4f}, NMI: {nmi:.4f}, ARI: {ari:.4f}')

        f1, sc = eva2(X_train, y, y_pred)
        print(f'FINAL RESULTS -> F1: {f1:.4f}, SC: {sc:.4f}')


    # ==========================================================
    # Extract and Save the Latent Space
    # ==========================================================
    import pandas as pd
    import os

    print("\n--- Saving Latent Space ---")
        
    # Define where to save the embeddings
    # It will save in the same folder as your model weights
    embedding_path = os.path.join(args.model_pth, "scG_latent_space.csv")
    labels_path = os.path.join(args.model_pth, "scG_predictions.csv")

    # Convert the latent space array (X_train) to a pandas DataFrame and save
    latent_df = pd.DataFrame(X_train)
    latent_df.to_csv(embedding_path, index=False, header=False)
        
    # Save the predicted cluster labels as well
    labels_df = pd.DataFrame(y_pred, columns=["Cluster"])
    labels_df.to_csv(labels_path, index=False)

