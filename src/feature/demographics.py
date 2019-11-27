import numpy as np
import scipy.sparse as sps
import pandas as pd
from src.feature.clustering_utils import cluster_data

def get_sub_class_demographic():
    raise NotImplemented()

def get_asset_demographic():
    raise NotImplemented()

def get_price_demographic(URM_train, ICM):
    raise NotImplemented()

def get_clustering_item_demographic():
    raise NotImplemented()

def get_clustering_user_demographic(dataframe: pd.DataFrame, bins, n_init, init_method="Huang", seed=69420):
    """
    Return a user-demographic based on a clustering approach.
    Users are clustered according to the data frame containing information about URM profile length and UCM.

    The default parameters are the ones that are given by the clustering achieving the best results in term
    of cost function considering the default number of bins, which is 10.

    :param dataframe: mixed dataframe of URM-profile length and information contained in the UCM
    :param bins: number of bins
    :param n_init: number of initialization of the cluster
    :param seed: clustering approach seed
    :return: clusters of the data, and a list containing the cluster ids
    """
    clusters = cluster_data(dataframe, n_clusters=bins, n_init=n_init, init_method=init_method, seed=seed)
    cluster_id_list = np.arange(bins).tolist()

    return clusters, cluster_id_list

def get_user_profile_demographic(URM_train, bins):
    """
    Return the user profiles demographic of the URM_train given, splitting equally the bins.

    :param URM_train: URM used for training the given recommender
    :param bins: number of bins
    :return: user profile demographics described as a tuple containing (size of the block, profile lengths,
    user sorted by the the profile length, mean of each group
    """
    # Building user profiles groups
    URM_train = sps.csr_matrix(URM_train)
    profile_length = np.ediff1d(URM_train.indptr) # Getting the profile length for each user
    sorted_users = np.argsort(profile_length) # Arg-sorting the user on the basis of their profiles len
    block_size = int(len(profile_length) * (1/bins)) # Calculating the block size, given the desired number of bins

    group_mean_len = []

    # Print some stats. about the bins
    for group_id in range(0, bins):
        start_pos = group_id * block_size
        if group_id < bins - 1:
            end_pos = min((group_id + 1) * block_size, len(profile_length))
        else:
            end_pos = len(profile_length)
        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        group_mean_len.append(int(users_in_group_p_len.mean()))

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

    return block_size, profile_length, sorted_users, group_mean_len
