import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.feature.demographics_content import get_sub_class_content
from src.utils.general_utility_functions import get_split_seed

# SETTINGS
KEEP_OUT = 1
SAVE_ON_FILE = False
PRINT_SINGLE_SUBCLASS = False
PRINT_AT_LEAST_50 = True


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
    # Path creation
    if SAVE_ON_FILE:
        version_path = "../../report/graphics/exploration/subclass_diversity/"
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

    ICM_subclass = data_reader.get_ICM_from_name("ICM_sub_class")
    subclass_feature_to_id_mapper = data_reader.dataReader_object.get_ICM_feature_to_index_mapper_from_name(
        "ICM_sub_class")
    subclass_content_dict = get_sub_class_content(ICM_subclass, subclass_feature_to_id_mapper, binned=True)
    subclass_content = get_sub_class_content(ICM_subclass, subclass_feature_to_id_mapper, binned=False)

    user_act = (URM_train > 0).sum(axis=1)
    user_act = np.array(user_act).squeeze()

    if PRINT_SINGLE_SUBCLASS:
        write_file_and_print("ALL USERS THAT LIKED ONLY A SUBCLASS \n\n", f)
        for user in range(URM_train.shape[0]):
            items_liked = URM_train[user].indices
            subclass_item_liked = subclass_content[items_liked]

            if np.unique(subclass_item_liked).size == 1 and user_act[user] > 1:
                write_file_and_print("User {} - profile {}".format(user, user_act[user]), f)

    if PRINT_AT_LEAST_50:
        write_file_and_print("USERS THAT HAS AT LEAST 50% OF INTERACTIONS IN A SUBCLASS \n\n", f)
        for user in range(URM_train.shape[0]):
            items_liked = URM_train[user].indices
            subclass_item_liked = subclass_content[items_liked]

            counts = np.bincount(subclass_item_liked)
            most_liked_subclass = np.argmax(counts)

            mask = subclass_item_liked == most_liked_subclass

            if user_act[user] > 7 and subclass_item_liked[mask].size > 0.5 * subclass_item_liked.size:
                write_file_and_print("User {} - profile {}".format(user, user_act[user]), f)
