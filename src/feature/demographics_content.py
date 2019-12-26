import numpy as np
import scipy.sparse as sps
import pandas as pd

from src.data_management.data_getter import get_warmer_UCM
from src.feature.clustering_utils import cluster_data


def get_sub_class_content(ICM_subclass, original_feature_to_id_mapper, binned=True):
    """
    Return a list containing all items in the different subclass

    Note: we should however, that position cannot be used to things correctly, since there are some missing wholes
    Therefore, the mapper should be always taken into account

    :param binned: True if you want a dictionary, false otherwise
    :param original_feature_to_id_mapper:
    :param ICM_subclass: ICM of the subclasses
    :return: if binned a dictionary containing arrays for each key: the items of that subclass,
    else we will have subclass content
    """
    ICM_copy = ICM_subclass.tocoo(copy=True)

    items = ICM_copy.row
    features = ICM_copy.col
    id_to_original_mapper = {v: int(k.split("-")[1]) for k, v in original_feature_to_id_mapper.items()}
    original_features = [id_to_original_mapper[id_value] for id_value in features]
    subclass_content = np.full(ICM_copy.shape[0], -1)
    subclass_content[items] = original_features

    if binned:
        new_dict = {}
        for f in np.unique(original_features):
            # Find items with that subclass
            items = np.argwhere(subclass_content == f).squeeze()
            new_dict[f] = items

        return new_dict
    else:
        return subclass_content

def get_asset_demographic(ICM_asset, bins):
    raise NotImplemented()


def get_price_demographic(ICM_price, bins):
    raise NotImplemented()


def get_clustering_item_demographic():
    raise NotImplemented()


def get_user_demographic(UCM, original_feature_to_id_mapper, binned=False):
    """
    Return a list containing all demographics with only users that has profile length more than threshold_users.
    In case there is no demographic for that user, it returns -1
     - This is useful for plotting the metric based on age demographic

     So, we can call this for the purpose of getting user demographic of age and region.

    :param original_feature_to_id_mapper:
    :param UCM: any UCM age or region
    :param binned: true if you want to obtain these demographics in an already grouped way. In which, in each
    list, we have users from the same region/age/etc., and so on. The number of bins is determined automatically,
    since the possible features present (region/age) are already discretized)
    :return: a list containing all demographics with only users that has profile length more than threshold_users
    """
    UCM_copy = UCM.tocoo(copy=True)

    users = UCM_copy.row
    features = UCM_copy.col
    id_to_original_mapper = {v: int(k) for k, v in original_feature_to_id_mapper.items()}
    original_features = [id_to_original_mapper[id_value] for id_value in features]
    user_demographic = np.full(UCM_copy.shape[0], -1)
    user_demographic[users] = original_features

    if binned:
        unique_features = np.unique(user_demographic)
        unique_features = np.sort(unique_features)
        result = []
        for i in unique_features:
            result.append(np.where(user_demographic == i)[0])
        user_demographic = result

    return user_demographic


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


def get_profile_demographic_wrapper(URM_train, bins, users=True):
    """
    Wrapper to be called if you wish to

    :param URM_train:
    :param bins: number of bins
    :param users: true if profile len of users is considered, false for item popularity
    :return: list of list: in each list, users/items in that cluster are present. Moreover,
    it also returns a list containing the identifiers of each group
    """
    block_size, profile_length, sorted_elems, group_mean_len = get_profile_demographic(URM_train, bins, users)

    clusters = []

    for group_id in range(0, bins):
        start_pos = group_id * block_size
        if group_id < bins - 1:
            end_pos = min((group_id + 1) * block_size, len(profile_length))
        else:
            end_pos = len(profile_length)
        elems_in_group = sorted_elems[start_pos:end_pos]
        elems_in_group_p_len = profile_length[elems_in_group]

        group_mean_len.append(int(elems_in_group_p_len.mean()))

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      elems_in_group_p_len.mean(),
                                                                      elems_in_group_p_len.min(),
                                                                      elems_in_group_p_len.max()))

        clusters.append(elems_in_group)
    return clusters, group_mean_len


def get_profile_demographic(URM_train, bins, users=True):
    """
    Return the user profiles demographic of the URM_train given, splitting equally the bins.

    :param URM_train: URM used for training the given recommender
    :param bins: number of bins
    :param users: true if profile len of users is returned, false for items
    :return: user profile demographics described as a tuple containing (size of the block, profile lengths,
    user sorted by the the profile length, mean of each group
    """
    # Building user profiles groups
    if users:
        URM_train = sps.csr_matrix(URM_train).copy()
    else:
        URM_train = sps.csc_matrix(URM_train).copy()
    profile_length = np.ediff1d(URM_train.indptr)  # Getting the profile length for each element
    sorted_elems = np.argsort(profile_length)  # Arg-sorting the elems on the basis of their profiles len
    block_size = int(len(profile_length) * (1 / bins))  # Calculating the block size, given the desired number of bins

    group_mean_len = []

    # Print some stats. about the bins
    for group_id in range(0, bins):
        start_pos = group_id * block_size
        if group_id < bins - 1:
            end_pos = min((group_id + 1) * block_size, len(profile_length))
        else:
            end_pos = len(profile_length)
        elems_in_group = sorted_elems[start_pos:end_pos]

        elems_in_group_p_len = profile_length[elems_in_group]

        group_mean_len.append(int(elems_in_group_p_len.mean()))

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      elems_in_group_p_len.mean(),
                                                                      elems_in_group_p_len.min(),
                                                                      elems_in_group_p_len.max()))

    return block_size, profile_length, sorted_elems, group_mean_len
