import numpy as np
import pandas as pd
from kmodes.kmodes import KModes


def cluster_data(data, clustering_method="kmodes", n_clusters=5, init_method="Huang", n_init=5, verbose=1, seed=69420, n_jobs=1):
    """
    Cluster data according to the given parameters.

    :param data: data to be clustered
    :param clustering_method: method to be used. The only available so far is "kmodes"
    :param n_clusters: number of clusters
    :param init: initialization method of the clustering algorithm
    :param n_init: number of trial in the clustering step
    :param verbose: if the output of the clustering should be verbose or not
    :param seed: fix the random state for the clustering method
    :param n_jobs
    :return: List of lists. Each list, is the list of users in the i-th cluster
    """
    # Clustering items
    if clustering_method == "kmodes":
        km = KModes(n_clusters=n_clusters, init=init_method, n_init=n_init, verbose=verbose, random_state=seed,
                    n_jobs=n_jobs)
    else:
        raise NotImplemented()

    clusters = km.fit_predict(data)

    # Get users clustered
    users_clustered_list = []
    for i in range(0, n_clusters):
        users_clustered_list.append(np.where(clusters == i)[0])

    return users_clustered_list


def preprocess_data_frame(df: pd.DataFrame, mapper_dict):
    """
    Pre-process the data frame removing the users discarded from the split in train-test.
    Then, change the index of the data frame (i.e. users number) according to mapper of the split.

    :param df: data frame read from file
    :param mapper_dict: dictionary mapping between original users, and then one in the split
    :return: preprocessed data frame
    """
    df = df.copy()

    # Here, we have the original users, we should map them to the user in the split
    warm_user_original = df.index
    keys = np.array(list(mapper_dict.keys()))  # Users that are actually mapped to something else

    mask = np.in1d(warm_user_original, keys)
    not_mask = np.logical_not(mask)
    user_to_discard = warm_user_original[not_mask]  # users that should be removed from the data frame since they

    # Removing users discarded from the data frame
    new_df = df.drop(user_to_discard, inplace=False)

    # Mapping the users
    new_df = new_df.reset_index()
    new_df['user'] = new_df['user'].map(lambda x: mapper_dict[str(x)])
    new_df = new_df.set_index("user")

    return new_df
