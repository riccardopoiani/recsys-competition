import numpy as np
import scipy.sparse as sps

def get_bootstrap_URM(URM_train):
    """
    Return a bootstrap of URM in csr matrix

    :param URM_train: URM to get the bootstrap from
    :return: bootstrap of URM in csr matrix
    """
    URM_copy = URM_train.tocoo(copy=True)
    row = URM_copy.row
    col = URM_copy.col
    data = URM_copy.data

    interactions_list = np.arange(0, len(data))
    sample_interactions_list = np.random.choice(interactions_list, len(data), replace=True)
    unique_sample, counts_sample = np.unique(sample_interactions_list, return_counts=True)
    sample_row = row[unique_sample]
    sample_col = col[unique_sample]
    sample_data = counts_sample
    URM_sample = sps.coo_matrix(URM_copy.shape)
    URM_sample.row = sample_row
    URM_sample.col = sample_col
    URM_sample.data = sample_data
    return URM_sample.tocsr()