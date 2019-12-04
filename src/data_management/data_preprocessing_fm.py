from scipy.sparse import coo_matrix
import numpy as np


def format_URM_negative_sampling_user_compressed(URM):
    pass


def format_URM_negative_sampling_non_compressed(URM):
    pass


def format_URM_positive_user_compressed(URM):
    """
    Format positive interactions of an URM in the way that is needed for the FM model.
    Here, however, users information are grouped w.r.t. items, meaning that, we will have:
    - We have #warm_items @row
    - We have #users+items+1 @cols
    - We have #(interactions)+(warm_items*2) @data

    Each row is representing a warm item and all users that interacted with that item are stored in that row.

    :param URM: URM to be preprocessed
    :return: preprocessed URM in sparse matrix csr format
    """

    warm_items_mask = np.ediff1d(URM.tocsc().indptr) > 0
    warm_items = np.arange(URM.shape[1])[warm_items_mask]

    new_train = URM.copy().tocoo()
    fm_matrix = coo_matrix((warm_items.size, URM.shape[0] + URM.shape[1] + 1), dtype=np.int8)

    # Index offset
    item_offset = URM.shape[0]

    # Set up initial vectors
    row_v = np.zeros(new_train.data.size + (warm_items.size * 2))
    col_v = np.zeros(new_train.data.size + (warm_items.size * 2))
    data_v = np.zeros(new_train.data.size + (warm_items.size * 2))  # Already ok, nothing to be added

    # For all the items, set up its content
    j = 0  # Index to scan and modify the vectors
    URM_train_csc = URM.copy().tocsc()
    for i, item in enumerate(warm_items):
        # Find all users who liked that item
        users_who_liked_item = URM_train_csc[:, item].indices
        offset = users_who_liked_item.size
        if offset > 0:
            col_v[j:j + offset] = users_who_liked_item
            row_v[j:j + offset] = i
            data_v[j:j + offset] = 1

            col_v[j + offset] = item + item_offset
            row_v[j + offset] = i
            data_v[j + offset] = 1

            col_v[j + offset + 1] = fm_matrix.shape[1] - 1
            row_v[j + offset + 1] = i
            data_v[j + offset + 1] = 1

            j = j + offset + 2
        else:
            raise RuntimeError("Illegale state")

    # Setting new information
    fm_matrix.row = row_v
    fm_matrix.col = col_v
    fm_matrix.data = data_v

    return fm_matrix.tocsr()


def format_URM_positive_non_compressed(URM):
    """
    Format positive interactions of an URM in the way that is needed for the FM model.
    - We have #num_ratings row
    - The last column with all the ratings (for implicit dataset it just a col full of 1
    - In each row there are 3 interactions: 1 for the user, 1 for the item, and 1 for the rating
    - Only positive samples are encoded here

    Note: this method works only for implicit dataset

    :param URM: URM to be preprocessed
    :return: csr_matrix containing the URM preprocessed in the described way
    """
    new_train = URM.copy().tocoo()
    fm_matrix = coo_matrix((URM.data.size, URM.shape[0] + URM.shape[1] + 1), dtype=np.int8)

    # Index offset
    item_offset = URM.shape[0]

    # Last col
    last_col = URM.shape[0] + URM.shape[1]

    # Set up initial vectors
    row_v = np.zeros(new_train.data.size * 3)  # Row should have (i,i,i) repeated for all the size
    col_v = np.zeros(new_train.data.size * 3)  # This is the "harder" to set
    data_v = np.ones(new_train.data.size * 3)  # Already ok, nothing to be added

    # Setting row vector
    for i in range(0, new_train.data.size):
        row_v[3 * i] = i
        row_v[(3 * i) + 1] = i
        row_v[(3 * i) + 2] = i

    # Setting col vector
    for i in range(0, new_train.data.size):
        # Retrieving information
        user = new_train.row[i]
        item = new_train.col[i]

        # Fixing col indices to be added to the new matrix
        user_index = user
        item_index = item + item_offset

        col_v[3 * i] = user_index
        col_v[(3 * i) + 1] = item_index
        col_v[(3 * i) + 2] = last_col

    # Setting new information
    fm_matrix.row = row_v
    fm_matrix.col = col_v
    fm_matrix.data = data_v

    return fm_matrix.tocsr()
