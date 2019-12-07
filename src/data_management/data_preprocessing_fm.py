from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
import numpy as np
import scipy.sparse as sps

from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix


def mix_URM(URM_positive: csr_matrix, URM_negative: csr_matrix):
    return sps.vstack([URM_positive, URM_negative], format='csr')


def format_URM_slice_uncompressed(users, items_per_users, max_user_id, n_cols):
    fm_matrix_builder = IncrementalSparseMatrix(n_cols=n_cols)
    row_list = np.repeat(np.arange(items_per_users.shape[0] * items_per_users.shape[1]), repeats=2)
    col_list = np.zeros(shape=items_per_users.shape[0] * items_per_users.shape[1] * 2)
    user_col_list = np.repeat(users, repeats=items_per_users.shape[1])
    items_col_list = np.array(items_per_users).flatten() + max_user_id
    col_list[np.arange(items_per_users.shape[0] * items_per_users.shape[1]) * 2] = user_col_list
    col_list[np.arange(items_per_users.shape[0] * items_per_users.shape[1]) * 2 + 1] = items_col_list
    fm_matrix_builder.add_data_lists(row_list_to_add=row_list, col_list_to_add=col_list,
                                     data_list_to_add=np.ones(len(row_list)))
    return fm_matrix_builder.get_SparseMatrix()


#############################################################################
########################## INTEGRATING EXTERNAL INFORMATION##################
#############################################################################

def add_UCM_info(fm_matrix: csr_matrix, UCM: csr_matrix, user_offset):
    """
    Given a matrix in the format needed to FM, it adds information concerning the UCM

    Note: no group by items should be applied in this case

    :param fm_matrix: matrix containing dataset for FM models (last column has no rating list)
    :param UCM: UCM information about users
    :param user_offset: starting column index for items in fm_matrix (should be 0)
    :return: new matrix containing also information about the UCM
    """
    fm_matrix_copy = fm_matrix.copy()
    user_fm_matrix = fm_matrix[:, user_offset: user_offset + UCM.shape[0]].copy()
    UCM_fm_matrix = user_fm_matrix.dot(UCM)
    merged_fm = sps.hstack([fm_matrix_copy, UCM_fm_matrix], format="csr")
    return merged_fm


def add_ICM_info(fm_matrix: csr_matrix, ICM: csr_matrix, item_offset):
    """
    Given a matrix in the format needed for FM, it adds information concerning the ICM

    Note: no group by users should be applied in this case

    :param fm_matrix: matrix concerning dataset for FM models (last column has no rating list)
    :param ICM: ICM information about items
    :param item_offset: starting column index for items in fm_matrix (it should be URM_train.shape[0]
                        of the URM used to construct the fm_matrix)
    :return: new matrix integrating ICM data
    """
    fm_matrix_copy = fm_matrix.copy()
    item_fm_matrix = fm_matrix[:, item_offset: item_offset + ICM.shape[0]].copy()
    ICM_fm_matrix = item_fm_matrix.dot(ICM)
    merged_fm = sps.hstack([fm_matrix_copy, ICM_fm_matrix], format="csr")
    return merged_fm


#################################################################################
########################## SAMPLING STRATEGIES ##################################
#################################################################################
def uniform_sampling_strategy(negative_sample_size, URM, check_replacement=False):
    """
    Sample negative samples uniformly from the given URM

    :param negative_sample_size: number of negative samples to be sampled
    :param URM: URM from which samples are taken
    :param check_replacement: whether to check for replacement or not. Checking is expensive
    :return: bi-dimensional array of shape (2, negative_sample_size): in the first dimensions row-samples are
    stored, while in the second one col-samples are stored. Therefore, in the i-th col of this returned array
    you can find a indices of a negative sample in the URM_train
    """
    max_row = URM.shape[0]
    max_col = URM.shape[1]
    collected_samples = np.zeros(shape=(2, negative_sample_size))
    sampled = 0
    while sampled < negative_sample_size:
        if sampled % 10000 == 0:
            print("Sampled {} on {}".format(sampled, negative_sample_size))
        t_row = np.random.randint(low=0, high=max_row, size=1)[0]
        t_col = np.random.randint(low=0, high=max_col, size=1)[0]
        t_sample = np.array([[t_row], [t_col]])

        if check_replacement:
            if (not np.equal(collected_samples, t_sample).min(axis=0).max()) and (URM[t_row, t_col] == 0):
                collected_samples[:, sampled] = [t_row, t_col]
                sampled += 1
        else:
            if URM[t_row, t_col] == 0:
                collected_samples[:, sampled] = [t_row, t_col]
                sampled += 1
    return collected_samples if check_replacement else np.unique(collected_samples, axis=1)


#################################################################################
########################## NEGATIVE RATING PREPARATION ##########################
#################################################################################

def format_URM_negative_sampling_user_compressed(URM: csr_matrix, negative_rate=1, check_replacement=False,
                                                 sampling_function=None):
    """
    Format negative interactions of an URM in the way that is needed for the FM model. Here, however, users
    and compressed w.r.t. the items they liked in the negative samples sampled

    In particular you will have:
    - #different_items_sampled @row
    - #users+items+1 @cols
    - #(negative_sample_size)*(different_items_sampled*2) @data

    :param URM: URM to be preprocessed  and from which negative samples are taken
    :param negative_rate: how much negatives samples do you want in proportion to the negative one
    :param check_replacement: whether to check for replacement or not. Checking costs time
    :param sampling_function: sampling function that takes in input the negative sample size
    and the URM from which samples are taken. If None, uniform sampling will be applied
    :return: csr_matrix containing the negative interactions:
    """
    negative_sample_size = int(URM.data.size * negative_rate)
    new_train = URM.copy().tocoo()
    item_offset = URM.shape[0]

    print("Start sampling...")

    if sampling_function is None:
        collected_samples = uniform_sampling_strategy(negative_sample_size=negative_sample_size, URM=URM,
                                                      check_replacement=check_replacement)
    else:
        collected_samples = sampling_function(negative_sample_size=negative_sample_size, URM=URM,
                                              check_replacement=check_replacement)
    # Different items sampled
    different_items_sampled = np.unique(collected_samples[1])

    fm_matrix = coo_matrix((different_items_sampled.size, URM.shape[0] + URM.shape[1] + 1), dtype=np.int8)

    row_v = np.zeros(new_train.data.size + (different_items_sampled.size * 2))
    col_v = np.zeros(new_train.data.size + (different_items_sampled.size * 2))
    data_v = np.zeros(new_train.data.size + (different_items_sampled.size * 2))

    print("Matrix builiding...", end="")

    # For all the items, set up its content
    j = 0  # Index to scan and modify the vectors
    URM_train_csc = URM.copy().tocsc()
    for i, item in enumerate(different_items_sampled):
        # Find all users sampled for that item
        item_mask = collected_samples[1] == item
        users_sampled_for_that_item = np.unique(collected_samples[0][item_mask])

        offset = users_sampled_for_that_item.size
        if offset > 0:
            col_v[j:j + offset] = users_sampled_for_that_item
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
            raise RuntimeError("Illegal state")

    print("Done")

    # Setting new information
    fm_matrix.row = row_v
    fm_matrix.col = col_v
    fm_matrix.data = data_v

    return fm_matrix.tocsr()


def format_URM_negative_sampling_non_compressed(URM: csr_matrix, negative_rate=1,
                                                sampling_function=None, check_replacement=False):
    """
    Format negative interactions of an URM in the way that is needed for the FM model
    - We have #positive_interactions * negative_rate @rows
    - We have #users+items+1 @cols
    - We have 3 interactions in each row: one for the users, one for the item, and -1 for the rating

    :param URM: URM to be preprocessed and from which negative samples is taken
    :param check_replacement: whether to check for replacement while sampling or not
    :param negative_rate: how much negatives samples do you want in proportion to the negative one
    :param sampling_function: sampling function that takes in input the negative sample size
    and the URM from which samples are taken (and if you want to check for replacement).
    If None, uniform sampling will be applied
    :return: csr_matrix containing the negative interactions
    """
    # Initial set-up
    item_offset = URM.shape[0]
    last_col = URM.shape[0] + URM.shape[1]
    negative_sample_size = int(URM.data.size * negative_rate)

    print("Start sampling...")

    # Take samples
    if sampling_function is None:
        collected_samples = uniform_sampling_strategy(negative_sample_size=negative_sample_size,
                                                      URM=URM, check_replacement=check_replacement)
    else:
        collected_samples = sampling_function(negative_sample_size=negative_sample_size, URM=URM,
                                              check_replacement=check_replacement)

    fm_matrix = coo_matrix((negative_sample_size, URM.shape[0] + URM.shape[1] + 1), dtype=np.int8)
    negative_sample_size = collected_samples[0].size

    # Set up initial vectors
    row_v = np.zeros(negative_sample_size * 3)  # Row should have (i,i,i) repeated for all the size
    col_v = np.zeros(negative_sample_size * 3)  # This is the "harder" to set
    data_v = -np.ones(negative_sample_size * 3)  # Already ok, nothing to be added

    print("Set up row of COO...")

    # Setting row vector
    for i in range(0, negative_sample_size):
        row_v[3 * i] = i
        row_v[(3 * i) + 1] = i
        row_v[(3 * i) + 2] = i

    print("Set up col of COO...")

    # Setting col vector
    for i in range(0, negative_sample_size):
        # Retrieving information
        user = collected_samples[0, i]
        item = collected_samples[1, i]

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


#################################################################################
########################## POSITIVE RATING PREPARATION ##########################
#################################################################################

def format_URM_positive_user_compressed(URM: csr_matrix):
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
            raise RuntimeError("Illegal state")

    # Setting new information
    fm_matrix.row = row_v
    fm_matrix.col = col_v
    fm_matrix.data = data_v

    return fm_matrix.tocsr()


def format_URM_positive_non_compressed(URM: csr_matrix):
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
