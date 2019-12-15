import numpy as np
import scipy.sparse as sps

from src.utils.general_utility_functions import get_split_seed


def get_bootstrap_URM(URM_train, weight_replacement):
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
    np.random.seed(get_split_seed())
    sample_interactions_list = np.random.choice(interactions_list, len(data), replace=True)
    np.random.seed()
    unique_sample, counts_sample = np.unique(sample_interactions_list, return_counts=True)
    URM_sample = sps.coo_matrix(URM_copy.shape)
    URM_sample.row = row[unique_sample]
    URM_sample.col = col[unique_sample]
    URM_sample.data = data[unique_sample]
    if weight_replacement:
        counts_mask = counts_sample > 1
        replace_sample = unique_sample[counts_mask]
        replace_counts = counts_sample[counts_mask] - 1
        samples = np.repeat(replace_sample, replace_counts)
        new_users = row[samples]
        unique_users, counts_users = np.unique(new_users, return_counts=True)
        new_unique_users_id = np.arange(len(unique_users))
        new_users = np.repeat(new_unique_users_id, counts_users)
        URM_replacement = sps.coo_matrix((len(new_unique_users_id), URM_copy.shape[1]))
        URM_replacement.row = new_users
        URM_replacement.col = col[samples]
        URM_replacement.data = data[samples]
        URM_sample = sps.vstack([URM_sample, URM_replacement])

    return URM_sample.tocsr()


def get_user_bootstrap(URM_train):
    URM_copy = URM_train.tocsr(copy=True)
    index_pointer = URM_copy.indptr
    col = URM_copy.indices
    data = URM_copy.data

    user_list = np.arange(URM_copy.shape[0])
    np.random.seed(get_split_seed())
    sample_user_list = np.random.choice(user_list, URM_copy.shape[0], replace=True)
    np.random.seed()
    unique_sample, first_index_sample, counts_sample = np.unique(sample_user_list, return_counts=True,
                                                                 return_index=True)
    start = index_pointer[unique_sample]
    end = index_pointer[unique_sample+1]
    n_elements = end - start
    indices = n_ranges(start, end)

    URM_sample = sps.coo_matrix(URM_copy.shape)
    URM_sample.row = np.repeat(unique_sample, n_elements)
    URM_sample.col = col[indices]
    URM_sample.data = data[indices]

    replace_mask = np.full(len(sample_user_list), fill_value=True, dtype=np.bool)
    replace_mask[first_index_sample] = False
    replace_sample = sample_user_list[replace_mask]
    URM_replace = URM_copy[replace_sample, :]
    URM_sample = sps.vstack([URM_sample, URM_replace], format="csr")
    return URM_sample


def n_ranges(start, end, return_flat=True):
    """
    Returns n ranges, n being the length of start (or end, they must be the same length) where each value in
    start represents the start of a range, and a value in end at the same index the end of it

    :param start: 1D np.ndarray representing the start of a range. Each value in start must be <=
                  than that of stop in the same index
    :param end: 1D np.ndarray representing the end of a range
    :param return_flat:
    :return All ranges flattened in a 1darray if return_flat is True otherwise an array of arrays with a range in each
    """
    # lengths of the ranges
    lens = end - start
    # repeats starts as many times as lens
    start_rep = np.repeat(start, lens)
    # helper mask with as many True in each row
    # as value in same index in lens
    arr = np.arange(lens.max())
    m =  arr < lens[:,None]
    # ranges in a flattened 1d array
    # right term is a cumcount up to each len
    ranges = start_rep + (arr * m)[m]
    # returns if True otherwise in split arrays
    if return_flat:
        return ranges
    else:
        return np.split(ranges, np.cumsum(lens)[:-1])