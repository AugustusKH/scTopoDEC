import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

# Aliases for easier calling in the main script
ari = adjusted_rand_score
nmi = normalized_mutual_info_score

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy using the Hungarian algorithm.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    # Check that dimensions match
    assert y_pred.size == y_true.size, "Predictions and Truth must be the same length!"

    # 1. Compute the contingency matrix instantly (replaces the slow for-loop)
    w = contingency_matrix(y_true, y_pred)

    # 2. Run the Hungarian algorithm to find the optimal cluster mapping
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # 3. Calculate accuracy using fast NumPy vectorization
    accuracy = w[row_ind, col_ind].sum() / y_pred.size
    
    return accuracy