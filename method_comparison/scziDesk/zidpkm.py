import argparse
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
from scipy.optimize import linear_sum_assignment  # FIXED: Modern hungarian matching engine
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Pure TF2 imports matching our modernization pipeline
from preprocess import prepro, normalize
from network import ScziDeskAutoencoder  # Class name updated to match our TF2 file change
from utils import decode

logger = logging.getLogger(__name__)


def cluster_acc(y_true, y_pred):
    """
    Computes clustering accuracy using optimal Hungarian matching bipartite alignment.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    # FIXED: Replaced completely removed sklearn wrapper with standard scipy implementation
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


if __name__ == "__main__":
    random_seed = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 10000]

    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataname", default="Quake_10x_Trachea", type=str)
    parser.add_argument("--distribution", default="ZINB")
    parser.add_argument("--self_training", default=True, type=bool)
    parser.add_argument("--dims", default=[500, 256, 64, 32], type=list)
    parser.add_argument("--highly_genes", default=500, type=int)
    parser.add_argument("--alpha", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.001, type=float)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--random_seed", default=random_seed)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--update_epoch", default=10, type=int)
    parser.add_argument("--pretrain_epoch", default=1000, type=int)
    parser.add_argument("--funetrain_epoch", default=2000, type=int)
    parser.add_argument("--t_alpha", default=1.0, type=float)
    parser.add_argument("--noise_sd", default=1.5, type=float)
    parser.add_argument("--error", default=0.001, type=float)
    parser.add_argument("--gpu_option", default="0", type=str)

    args = parser.parse_args()

    # Data Intake Pipelines
    X, Y = prepro(args.dataname)
    X = np.ceil(X).astype(int)  # FIXED: np.int -> int
    count_X = X.copy()

    adata = sc.AnnData(X)
    adata.obs['Group'] = Y
    adata = normalize(adata, copy=True, highly_genes=args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    
    X = adata.X.astype(np.float32)
    Y = np.array(adata.obs["Group"])
    high_variable = np.array(adata.var.highly_variable.index, dtype=int)  # FIXED: np.int -> int
    count_X = count_X[:, high_variable]
    size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
    
    cluster_number = int(max(Y) - min(Y) + 1)
    result = []

    for seed in args.random_seed:
        # Step A: Enforce deterministic states across modern computation graphs
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.backend.clear_session()  # FIXED: Replaces old tf.reset_default_graph()

        # Step B: Instantiate our TF2 Subclassed Model
        chencluster = ScziDeskAutoencoder(
            dataname=args.dataname, 
            distribution=args.distribution, 
            self_training=args.self_training, 
            dims=args.dims, 
            cluster_num=cluster_number, 
            t_alpha=args.t_alpha,
            alpha=args.alpha, 
            gamma=args.gamma, 
            learning_rate=args.learning_rate, 
            noise_sd=args.noise_sd
        )
        
        # Step C: Optimization passes
        chencluster.pretrain(X, count_X, size_factor, args.batch_size, args.pretrain_epoch, args.gpu_option)
        chencluster.funetrain(X, count_X, size_factor, args.batch_size, args.funetrain_epoch, args.update_epoch, args.error)
        
        # Step D: Scoring Metrics Compilation
        kmeans_accuracy = np.around(cluster_acc(Y, chencluster.kmeans_pred), 5)
        kmeans_ARI = np.around(adjusted_rand_score(Y, chencluster.kmeans_pred), 5)
        kmeans_NMI = np.around(normalized_mutual_info_score(Y, chencluster.kmeans_pred), 5)
        
        accuracy = np.around(cluster_acc(Y, chencluster.Y_pred), 5)
        ARI = np.around(adjusted_rand_score(Y, chencluster.Y_pred), 5)
        NMI = np.around(normalized_mutual_info_score(Y, chencluster.Y_pred), 5)
        
        result.append([args.dataname, kmeans_accuracy, kmeans_ARI, kmeans_NMI, accuracy, ARI, NMI])

    # Convert results matrix into organized reporting dataframe
    output = pd.DataFrame(
        np.array(result), 
        columns=["dataset name", "kmeans accuracy", "kmeans ARI", "kmeans NMI", "accuracy", "ARI", "NMI"]
    )
    print(output)