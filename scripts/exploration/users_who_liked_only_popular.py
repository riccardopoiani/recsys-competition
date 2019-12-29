import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.utils.general_utility_functions import get_split_seed

# SETTINGS
KEEP_OUT = 1
SAVE_ON_FILE = True
POPULARITY_THRESHOLD = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PRINT_ACTIVITY = True
PRINT_NUMBER_OF_USERS = True


def write_file_and_print(s: str, file):
    print(s)
    if SAVE_ON_FILE:
        file.write(s)
        file.write("\n")
        file.flush()


def show_fig(name):
    fig = plt.gcf()
    fig.show()
    if SAVE_ON_FILE:
        new_file = output_folder_path + name + ".png"
        fig.savefig(new_file)


if __name__ == '__main__':
    """
    Comments:
    - All the users with the current popularity list that matches the conditions of having liked
    only very popular items are user with very few interactions 
    (e.g. 1, 2 or, rarely, 3). 
    - On these users, what are the performances of a top popular vs the performances of our current best models?
    Probably they are too few to have some decent information
    However, maybe, we can improve performances on these users, using the UserCBFCF recommender used for the cold users
    """
    # Path creation
    if SAVE_ON_FILE:
        version_path = "../../report/graphics/exploration/easy_users/"
        now = datetime.now().strftime('%b%d_%H-%M-%S')
        output_folder_path = version_path + now + "/"
        output_file_name = output_folder_path + "results.txt"
        try:
            if not os.path.exists(output_folder_path):
                os.mkdir(output_folder_path)
        except FileNotFoundError as e:
            os.makedirs(output_folder_path)

        f = open(output_file_name, "w")
    else:
        f = None

    # Data loading
    root_data_path = "../../data/"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Find users who liked only very popular items -> They should be very easy to recommend
    pop_items = (URM_train > 0).sum(axis=0)
    pop_items = np.array(pop_items).squeeze()
    pop_items = np.argsort(pop_items)  # item indices sorted by increasing popularity

    easy_users_per_threshold = []
    for threshold in POPULARITY_THRESHOLD:
        print("Collecting data on threshold {}".format(threshold))
        curr_list = []
        for user in range(URM_train.shape[0]):
            items_liked = URM_train[user].indices
            threshold_items = pop_items[-threshold:]  # threshold-most popular items

            mask_liked_threshold = np.in1d(items_liked, threshold_items)
            liked_in_threshold = items_liked[mask_liked_threshold]

            mask_threshold_liked = np.in1d(threshold_items, items_liked)
            threshold_in_liked = threshold_items[mask_threshold_liked]

            if liked_in_threshold.size == items_liked.size:
                curr_list.append(user)

        easy_users_per_threshold.append(curr_list)

    # What kind of users are they? What is their profile length? Are we performing for their recommendations?
    if PRINT_ACTIVITY:
        for i, easy_users in enumerate(easy_users_per_threshold):
            easy_users = np.array(easy_users)
            write_file_and_print("Threshold {} \n".format(POPULARITY_THRESHOLD[i]), f)
            for user in easy_users:
                # Is the user popular?
                activity = URM_train[user].indices.size
                write_file_and_print("Activity: {} \n".format(activity), f)

    if PRINT_NUMBER_OF_USERS:
        for i, easy_users in enumerate(easy_users_per_threshold):
            easy_users = np.array(easy_users)
            write_file_and_print("Threshold {} - User size {}".format(POPULARITY_THRESHOLD[i], easy_users.size), f)
