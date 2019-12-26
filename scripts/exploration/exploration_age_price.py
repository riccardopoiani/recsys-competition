import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train, get_users_of_age
from src.feature.demographics_content import get_user_demographic, get_sub_class_content
from src.utils.general_utility_functions import get_split_seed

# SETTINGS
KEEP_OUT = 1
SAVE_ON_FILE = False


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
        version_path = "../../report/graphics/exploration/age_subclass/{}/".format(AGE_TO_FOCUS)
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
    ICM_all = get_ICM_train(data_reader)
    UCM_all = get_UCM_train(data_reader)

    UCM_age = data_reader.get_UCM_from_name("UCM_age")
    age_feature_to_id_mapper = data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_age")
    age_demographic = get_user_demographic(UCM_age, age_feature_to_id_mapper, binned=True)

    ICM_price = data_reader.get_ICM_from_name("ICM_price")

    # DOES SUBCLASS DISTRIBUTION CHANGES BETWEEN AGES?
    # Collect distributions
    age_list = np.sort(np.array(list(age_feature_to_id_mapper.keys())))
    age_list = [int(age) for age in age_list]
    sub_age_dict = {}
    plt.title("Price distribution per age")
    plt.xlabel("Item index")
    plt.ylabel("Price")
    for age in age_list:
        users_age = get_users_of_age(age_demographic=age_demographic, age_list=[age])
        URM_age = URM_train[users_age].copy()
        items_age = URM_age.indices
        prices_ages = ICM_price[items_age].data
        prices_ages = np.sort(prices_ages)
        plt.plot(prices_ages, label="Age {}".format(age))

    plt.legend()
    show_fig("price_distribution_per_age")

    # Mean plot
    age_list.sort()
    mean_price_per_age = []
    for age in age_list:
        users_age = get_users_of_age(age_demographic=age_demographic, age_list=[age])
        URM_age = URM_train[users_age].copy()
        items_age = URM_age.indices
        prices_ages = ICM_price[items_age].data
        mean_price_per_age.append(prices_ages.mean())

    plt.title("Mean price per age")
    plt.xlabel("Age")
    plt.ylabel("Mean price")
    plt.plot(mean_price_per_age)
    show_fig("mean_price_per_age")
