def read_target_playlist(path = "../data/target_playlists.csv"):
    '''
    :return: list of user to recommend in the target playlist
    '''
    import numpy as np

    target_file = open(path, 'r')

    target_file.seek(0)
    target_tuple = []

    for line in target_file:
        if line != "playlist_id\n":
            target_tuple.append(row_split_target(line))

    return target_tuple



def read_user_rating_matrix():
    '''
    :return: all the user rating matrix, in csr format
    '''
    import scipy.sparse as sps
    import numpy as np

    # Reading data
    path = "../data/train.csv"
    user_rating_matrix_file = open(path, 'r')

    user_rating_matrix_file.seek(0)
    user_rating_matrix_tuples = []

    for line in user_rating_matrix_file:
        if line != "playlist_id,track_id\n":
            user_rating_matrix_tuples.append(row_split(line))

    # Creating URM
    user_list, item_list = zip(*user_rating_matrix_tuples)
    ones = np.ones(len(item_list))
    user_rating_matrix = sps.coo_matrix((ones, (user_list, item_list)))
    user_rating_matrix = user_rating_matrix.tocsr()

    return user_rating_matrix


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
