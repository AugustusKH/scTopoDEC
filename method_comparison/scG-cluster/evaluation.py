import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm.
    Optimized via SciPy and Scikit-Learn.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    # Shift labels to start from 0 safely
    y_true = y_true - np.min(y_true)
    y_pred = y_pred - np.min(y_pred)

    # 1. Fast cost matrix computation using a confusion matrix
    # Row = true class, Col = predicted class
    w = confusion_matrix(y_true, y_pred)

    # 2. Hungarian algorithm via SciPy
    # linear_sum_assignment finds the minimum cost, so we subtract from the max to find maximum overlap
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # Calculate optimal accuracy
    acc = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_true.size

    # 3. Create mapped predictions for F1 score (safely, without modifying the original y_pred)
    map_dict = {c: r for r, c in zip(row_ind, col_ind)}
    new_predict = np.array([map_dict.get(val, val) for val in y_pred])

    f1_macro = f1_score(y_true, new_predict, average='macro')
    
    return acc, f1_macro


def eva(y_true, y_pred, best_acc, best_nmi, best_ari, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    
    if nmi > best_nmi:
        best_acc = acc
        best_nmi = nmi
        best_ari = ari
        
    print(f"{epoch} :acc {acc:.4f} , nmi {nmi:.4f} , ari {ari:.4f} , f1 {f1:.4f}")
    return best_acc, best_nmi, best_ari


def eva1(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    
    print(f"{epoch} :acc {acc:.4f} , nmi {nmi:.4f} , ari {ari:.4f} , f1 {f1:.4f}")
    return acc, nmi, ari


def eva2(data, y_true, y_pred):
    # Only unpack what is needed
    _, f1 = cluster_acc(y_true, y_pred)
    sc = silhouette_score(data, y_pred)   # data = X_train

    return f1, sc