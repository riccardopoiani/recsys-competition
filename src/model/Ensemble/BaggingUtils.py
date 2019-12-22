import numpy as np
import scipy.sparse as sps

from src.utils.general_utility_functions import get_split_seed, n_ranges


def get_user_bootstrap(URM_train, seed):
    """
    Return a boostrap of URM_train by sampling users. The users sampled more than one times are stacked below the new
    sampled URM_train.

    :param URM_train: train URM to be sampled
    :return: a bootstrap of URM_train
    """
    rating_ui = URM_train.tocsr(copy=True)
    index_pointer = rating_ui.indptr
    col = rating_ui.indices
    data = rating_ui.data

    user_list = np.arange(rating_ui.shape[0])
    np.random.seed(seed)
    sample_user_list = np.random.choice(user_list, rating_ui.shape[0], replace=True)
    unique_sample, first_index_sample, counts_sample = np.unique(sample_user_list, return_counts=True,
                                                                 return_index=True)
    start = index_pointer[unique_sample]
    end = index_pointer[unique_sample+1]
    n_elements = end - start
    indices = n_ranges(start, end)

    rating_ui_sample = sps.coo_matrix(rating_ui.shape)
    rating_ui_sample.row = np.repeat(unique_sample, n_elements)
    rating_ui_sample.col = col[indices]
    rating_ui_sample.data = data[indices]

    replace_mask = np.full(len(sample_user_list), fill_value=True, dtype=np.bool)
    replace_mask[first_index_sample] = False
    replace_sample = sample_user_list[replace_mask]

    rating_ui_replace = rating_ui[replace_sample, :]
    rating_ui_sample = sps.vstack([rating_ui_sample, rating_ui_replace], format="csr")
    return rating_ui_sample, replace_sample


def get_item_bootstrap(URM_train, seed):
    """
    Return a boostrap of URM_train by sampling items. The items sampled more than one times are stacked on the right
    side of the new sampled URM_train.

    :param URM_train: train URM to be sampled
    :return: a bootstrap of URM_train
    """
    rating_iu = URM_train.T.tocsr(copy=True)
    index_pointer = rating_iu.indptr
    col = rating_iu.indices
    data = rating_iu.data

    item_list = np.arange(rating_iu.shape[0])
    np.random.seed(seed)  # Set seed to make the experiments reproducible
    sample_item_list = np.random.choice(item_list, rating_iu.shape[0], replace=True)
    unique_sample, first_index_sample, counts_sample = np.unique(sample_item_list, return_counts=True,
                                                                 return_index=True)

    start = index_pointer[unique_sample]
    end = index_pointer[unique_sample + 1]
    n_elements = end - start
    indices = n_ranges(start, end)

    rating_iu_sample = sps.coo_matrix(rating_iu.shape)
    rating_iu_sample.row = np.repeat(unique_sample, n_elements)
    rating_iu_sample.col = col[indices]
    rating_iu_sample.data = data[indices]*np.repeat(counts_sample, n_elements)

    """replace_mask = np.full(len(sample_item_list), fill_value=True, dtype=np.bool)
    replace_mask[first_index_sample] = False
    replace_sample = sample_item_list[replace_mask]

    rating_iu_replace = rating_iu[replace_sample, :]
    rating_iu_sample = sps.vstack([rating_iu_sample, rating_iu_replace], format="csr")"""
    return rating_iu_sample.T.tocsr(), 0