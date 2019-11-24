def read_target_users(path="../data/data_target_users_test.csv"):
    '''
    :return: list of user to recommend in the target playlist
    '''
    target_file = open(path, 'r')

    target_file.seek(0)
    target_tuple = []

    for line in target_file:
        if line != "user_id\n":
            target_tuple.append(row_split_target(line))

    return target_tuple

def read_URM_cold_all(path="../data/data_train.csv"):
    '''
    :return: all the user rating matrix, in csr format
    '''
    import scipy.sparse as sps
    import numpy as np
    import pandas as pd

    # Reading data
    df_original = pd.read_csv(path)

    user_id_list = df_original['row'].values
    item_id_list = df_original['col'].values
    rating_list = np.ones(len(user_id_list))

    # Creating URM
    URM_all = sps.coo_matrix((rating_list, (user_id_list, item_id_list)))
    URM_all = URM_all.tocsr()

    return URM_all

def row_split(row_string):
    '''
    Helper for splitting the URM
    :param row_string: line of the URM train
    :return: splitted row
    '''
    row_string = row_string.replace("\n", "")
    split = row_string.split(",")

    split[0] = int(split[0])
    split[1] = int(split[1])

    result = tuple(split)

    return result


def row_split_target(row_string):
    '''
    Function helper to read the target playlist
    :param row_string:
    :return:
    '''
    return int(row_string.replace("\n", ""))


def get_warm_user_rating_matrix(user_rating_matrix):
    '''
    :param user_rating_matrix: user rating matrix
    :return: warm version of the user rating matrix
    '''
    import numpy as np
    warm_items_mask = np.ediff1d(user_rating_matrix.tocsc().indptr) > 0
    warm_items = np.arange(user_rating_matrix.shape[1])[warm_items_mask]

    user_rating_matrix = user_rating_matrix[:, warm_items]

    warm_users_mask = np.ediff1d(user_rating_matrix.tocsr().indptr) > 0
    warm_users = np.arange(user_rating_matrix.shape[0])[warm_users_mask]

    user_rating_matrix = user_rating_matrix[warm_users, :]

    return user_rating_matrix
