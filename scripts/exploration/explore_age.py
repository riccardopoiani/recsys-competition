import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train, get_ignore_users_age
from src.feature.demographics_content import get_user_demographic
from src.utils.general_utility_functions import get_split_seed

# SETTINGS
AGE = 4
KEEP_OUT = 1
SAVE_ON_FILE = True
N_MOST_LIKED_ITEMS_TO_SHOW = 10


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
        version_path = "../../report/graphics/age_exploration/{}/".format(AGE)
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

    # Finding users with this age
    UCM_age = data_reader.get_UCM_from_name("UCM_age")
    age_feature_to_id_mapper = data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_age")
    age_demographic = get_user_demographic(UCM_age, age_feature_to_id_mapper, binned=True)
    users_oth_age = np.unique(get_ignore_users_age(age_demographic, [AGE]))
    total_users = np.arange(URM_train.shape[0])
    users_age = np.in1d(total_users, users_oth_age, invert=True)
    users_age = total_users[users_age]

    write_file_and_print("There are {} users of age {}".format(users_age.size, AGE), f)

    URM_train_age = URM_train[users_age].copy()

    # What are the number of interactions that we have for these users? What is their distribution?
    n_tot_interactions = URM_train_age.sum()
    n_avg_interactions = URM_train_age.sum(axis=1).mean()
    n_std_interactions = URM_train_age.sum(axis=1).std()
    write_file_and_print("There are {} total interactions for users of this age.\n"
                         "Avg number of interactions = {} \n"
                         "Std number of interactions = {} \n\n".format(n_tot_interactions, n_avg_interactions,
                                                                       n_std_interactions), f)

    interactions_per_user = np.squeeze(np.asarray(URM_train_age.sum(axis=1)))
    interactions_per_user = np.sort(interactions_per_user)

    plt.title("Number of interactions of users of age {}".format(AGE))
    plt.xlabel('User index')
    plt.ylabel('Number of interactions')
    plt.plot(interactions_per_user)
    show_fig("activity")

    # What is the item popularity among them? Do they like popular items?
    items_liked_age = np.squeeze(np.asarray(URM_train_age.sum(axis=0)))
    items_liked_age = np.sort(items_liked_age)
    items_liked_all = np.sort(np.squeeze(np.asarray(URM_train.sum(axis=0))))
    plt.title("Item popularity")
    plt.xlabel("Item index")
    plt.ylabel("Number of interactions")
    plt.plot(items_liked_age, label="Age {}".format(AGE))
    plt.plot(items_liked_all, label="All")
    plt.legend()
    show_fig("item_popularity")

    # What is the most liked item? How many interactions for it?
    item_indices = items_liked_age.argsort()[-N_MOST_LIKED_ITEMS_TO_SHOW:][::-1]
    for i in range(0, N_MOST_LIKED_ITEMS_TO_SHOW):
        n_most_liked = items_liked_age[item_indices[i]]
        write_file_and_print("The {}-th most liked item is {} with {} interactions. \n"
                             "Thus, it it is liked by {}% users. \n".format(i + 1, item_indices[i], n_most_liked,
                                                                            (n_most_liked / users_age.size) * 100), f)
    write_file_and_print("\n", f)

    if SAVE_ON_FILE:
        f.close()
