import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm for 
    optimal label matching. 
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    
    # We use (w.max() - w) because the algorithm finds the minimum cost,
    # and we want to find the maximum agreement.
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    # Build the confusion matrix
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size


def model_eval(adata, model_clusters, ground_truth):
    """
    Calculate clustering performane of models. 

    # Arguments
    adata : class:`anndata.AnnData`
        A single-cell object.
    model_clusters : `str`
        A key in `adata.obs` containing cluster labels predicted from models.
    ground_truth : `str`
        A key in `adata.obs` containing known cell-type or cluster labels.

    # Return
        clustering accuracy, NMI, ARI
    """
    y_pred = adata.obs[model_clusters].values
    y_true = adata.obs[ground_truth].values
    acc = np.round(cluster_acc(y_true, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), 5)
    #print(f"Clustering performance: ACC={acc}, NMI={nmi}, ARI={ari}")
    
    return acc, nmi, ari 