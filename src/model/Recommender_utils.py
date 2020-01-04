import numpy as np
import scipy.sparse as sps

from course_lib.Base.Recommender_utils import check_matrix


def apply_feature_weighting(matrix, feature_weighting="none"):
    from course_lib.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
    from course_lib.Base.Recommender_utils import check_matrix

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    if feature_weighting not in FEATURE_WEIGHTING_VALUES:
        raise ValueError(
            "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                FEATURE_WEIGHTING_VALUES, feature_weighting))

    if feature_weighting == "BM25":
        matrix = matrix.astype(np.float32)
        matrix = okapi_BM_25(matrix)
        matrix = check_matrix(matrix, 'csr')
    elif feature_weighting == "TF-IDF":
        matrix = matrix.astype(np.float32)
        matrix = TF_IDF(matrix)
        matrix = check_matrix(matrix, 'csr')
    return matrix


def userSimilarityMatrixTopK(user_weights, k=100):
    """
    The function selects the TopK most similar elements, column-wise

    :param user_weights:
    :param k:
    :return:
    """

    assert (user_weights.shape[0] == user_weights.shape[1]), "selectTopK: UserWeights is not a square matrix"

    n_users = user_weights.shape[1]
    k = min(k, n_users)

    # iterate over each column and keep only the top-k similar items
    data, rows_indices, cols_indptr = [], [], []

    user_weights = check_matrix(user_weights, format='csc', dtype=np.float32)

    for user_idx in range(n_users):

        cols_indptr.append(len(data))

        start_position = user_weights.indptr[user_idx]
        end_position = user_weights.indptr[user_idx + 1]

        column_data = user_weights.data[start_position:end_position]
        column_row_index = user_weights.indices[start_position:end_position]

        if min(k, column_data.size) == k:
            top_k_idx = np.argpartition(-column_data, kth=min(k, column_data.size-1))[:k]
        else:
            top_k_idx = np.ones(column_data.size, dtype=np.bool)

        data.extend(column_data[top_k_idx])
        rows_indices.extend(column_row_index[top_k_idx])

    cols_indptr.append(len(data))

    # During testing CSR is faster
    W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(n_users, n_users), dtype=np.float32)

    return W_sparse
