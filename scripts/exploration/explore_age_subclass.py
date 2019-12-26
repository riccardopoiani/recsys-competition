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
AGE_TO_FOCUS = 5
SAVE_ON_FILE = True
PLOT_GLOBAL_SUBCLASS_DISTR_PER_AGE = True
SUBCLASS_COUNT_THRESHOLD_LIST = [10, 50]
SUBCLASS_PERCENTAGE_MULTIPLIER_LIST = [10, 50, 100, 500, 1000]
PLOT_AGE_TO_FOCUS_VS_AGE = True
PLOT_AGE_TO_FOCUS_VS_ALL = True
PLOT_THRESHOLD_AGE_TO_FOCUS = True
PLOT_THRESHOLD_PERCENTAGE_AGE_TO_FOCUS = True


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

    ICM_subclass = data_reader.get_ICM_from_name("ICM_sub_class")
    subclass_feature_to_id_mapper = data_reader.dataReader_object.get_ICM_feature_to_index_mapper_from_name(
        "ICM_sub_class")
    subclass_content_dict = get_sub_class_content(ICM_subclass, subclass_feature_to_id_mapper, binned=True)
    subclass_content = get_sub_class_content(ICM_subclass, subclass_feature_to_id_mapper, binned=False)

    # DOES SUBCLASS DISTRIBUTION CHANGES BETWEEN AGES?
    # Collect distributions
    age_list = np.sort(np.array(list(age_feature_to_id_mapper.keys())))
    age_list = [int(age) for age in age_list]
    sub_age_dict = {}
    for age in age_list:
        users_age = get_users_of_age(age_demographic=age_demographic, age_list=[age])
        URM_age = URM_train[users_age].copy()
        items_age = URM_age.indices
        subclass_age = subclass_content[items_age]
        sub_age_dict[age] = subclass_age

    count_sub_ace_dict = {}
    for age in age_list:
        curr_count = np.zeros(subclass_content.max())
        curr_sub_age = sub_age_dict[age]
        for i in range(curr_count.size):
            mask = np.in1d(curr_sub_age, [i])
            curr_count[i] = curr_sub_age[mask].size

        count_sub_ace_dict[age] = curr_count
    count_global = np.zeros(count_sub_ace_dict[4].size)
    for elem in list(count_sub_ace_dict.values()):
        count_global += elem

    # Plot distributions
    if PLOT_GLOBAL_SUBCLASS_DISTR_PER_AGE:
        plt.title("Subclass distribution per age")
        plt.xlabel('Subclass index')
        plt.ylabel('Number of items liked per group')
        for age in age_list:
            plt.plot(count_sub_ace_dict[age], label="Age {}".format(age))
        plt.legend()
        show_fig("subclass_distribution")

    # Focus only on AGE_TO_FOCUS
    if PLOT_AGE_TO_FOCUS_VS_AGE:
        for age in age_list:
            if age != AGE_TO_FOCUS:
                plt.title("Age {} vs age {}".format(AGE_TO_FOCUS, age))
                plt.xlabel("Subclass index")
                plt.ylabel("Number of items liked per group")
                plt.plot(count_sub_ace_dict[AGE_TO_FOCUS], label="Age {}".format(AGE_TO_FOCUS))
                plt.plot(count_sub_ace_dict[age], label="Age {}".format(age))
                plt.legend()
                show_fig("age_{}_age_{}".format(AGE_TO_FOCUS, age))
    if PLOT_AGE_TO_FOCUS_VS_ALL:
        plt.title("Age {} vs global without {}".format(AGE_TO_FOCUS, AGE_TO_FOCUS))
        plt.xlabel("Subclass index")
        plt.ylabel("Number of items liked per group")
        plt.plot(count_global - count_sub_ace_dict[AGE_TO_FOCUS], label="All")
        plt.plot(count_sub_ace_dict[AGE_TO_FOCUS], label="Age {}".format(AGE_TO_FOCUS))
        plt.legend()
        show_fig("age_{}_vs_all".format(AGE_TO_FOCUS))

    # Subclass that users of AGE_TO_FOCUS didn't liked but that other groups have liked
    sub_age_to_focus = count_sub_ace_dict[AGE_TO_FOCUS]
    empty_subclass = np.argwhere(sub_age_to_focus == 0).squeeze()
    t_list = []
    for t in SUBCLASS_COUNT_THRESHOLD_LIST:
        curr_list = np.zeros(count_sub_ace_dict[AGE_TO_FOCUS].size)
        for sub in empty_subclass:
            if (count_global - count_sub_ace_dict[AGE_TO_FOCUS])[sub] > t:
                curr_list[sub] = 1
        t_list.append(curr_list)

    if PLOT_THRESHOLD_AGE_TO_FOCUS:
        for i, elem in enumerate(t_list):
            plt.title("Subclass with 0 ratings in users of age {}, "
                      "but with more than {} in others".format(AGE_TO_FOCUS, SUBCLASS_COUNT_THRESHOLD_LIST[i]))
            plt.xlabel("Subclass index")
            plt.ylabel("1 if condition is satifisfied, 0 otherwise")
            plt.plot(elem)
            show_fig("threshold_{}_age_{}".format(SUBCLASS_COUNT_THRESHOLD_LIST[i], AGE_TO_FOCUS))

    # Now, let's do the same thing, but instead of using 0, let's use percentage of ratings of users of AGE_TO_FOCUS
    # In particular, this should be useful to understand the tastes of the users
    total_ratings = URM_train.data.size
    total_ratings_age_to_focus = URM_train[get_users_of_age(age_demographic=age_demographic, age_list=[AGE_TO_FOCUS])].data.size
    percentage = total_ratings_age_to_focus / total_ratings
    t_list_above_percentage_per_threshold = []
    t_list_under_percentage_per_threshold = []

    for t in SUBCLASS_PERCENTAGE_MULTIPLIER_LIST:
        curr_list_upper = np.zeros(subclass_content.max())
        curr_list_lower = np.zeros(subclass_content.max())
        for sub in range(0, count_sub_ace_dict[AGE_TO_FOCUS].size):
            if count_global[sub] - count_sub_ace_dict[AGE_TO_FOCUS][sub] != 0:
                if count_sub_ace_dict[AGE_TO_FOCUS][sub] / (count_global - count_sub_ace_dict[AGE_TO_FOCUS])[sub] > t * percentage:
                    curr_list_upper[sub] = 1
            else:
                if count_sub_ace_dict[AGE_TO_FOCUS][sub] > 0:
                    curr_list_upper[sub] = 1

            if count_global[sub] - count_sub_ace_dict[AGE_TO_FOCUS][sub] != 0:
                if count_sub_ace_dict[AGE_TO_FOCUS][sub] / (count_global - count_sub_ace_dict[AGE_TO_FOCUS])[sub] < t * percentage:
                    curr_list_lower[sub] = 1
        t_list_above_percentage_per_threshold.append(curr_list_upper)
        t_list_under_percentage_per_threshold.append(curr_list_lower)

    if PLOT_THRESHOLD_PERCENTAGE_AGE_TO_FOCUS:
        for i, elem in enumerate(t_list_above_percentage_per_threshold):
            plt.title("Subclass with more ratings in {} * percentage for users with age {}"
                      .format(SUBCLASS_PERCENTAGE_MULTIPLIER_LIST[i], AGE_TO_FOCUS))
            plt.xlabel("Subclass index")
            plt.ylabel("1 if condition is satisfied, 0 otherwise")
            plt.plot(elem)
            show_fig("ut_perc_{}_age_{}".format(SUBCLASS_PERCENTAGE_MULTIPLIER_LIST[i], AGE_TO_FOCUS))

        for i, elem in enumerate(t_list_under_percentage_per_threshold):
            plt.title("Subclass with less ratings in {} * percentage for users with age {}"
                      .format(SUBCLASS_PERCENTAGE_MULTIPLIER_LIST[i], AGE_TO_FOCUS))
            plt.xlabel("Subclass index")
            plt.ylabel("1 if condition is satisfied, 0 otherwise")
            plt.plot(elem)
            show_fig("lt_perc_{}_age_{}".format(SUBCLASS_PERCENTAGE_MULTIPLIER_LIST[i], AGE_TO_FOCUS))

    if f is not None:
        f.close()