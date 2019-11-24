import numpy as np
import scipy.sparse as sps


def get_user_profile_demographic(URM_train, bins):
    """
    Return the user profiles demographic of the URM_train given, splitting equally the bins.

    :param URM_train:
    :param bins: number of bins
    :return: user profile demographics described as a tuple containing (size of the block, profile lenghts,
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
