import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from course_lib.Base.Similarity.Compute_Similarity import Compute_Similarity
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.feature.demographics_content import get_user_demographic
from src.utils.general_utility_functions import get_split_seed
from src.model.NewRatingMatrix import get_age_rating_matrix

# SETTINGS
SIMILARITY_TYPE = "cosine"
KEEP_OUT = 1
SAVE_ON_FILE = False
PLOT_SIMILARITY_MATRIX = False
PLOT_SINGLE_GRAPHICS = False


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
        version_path = "../../report/graphics/ARM/{}/".format(SIMILARITY_TYPE)
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

    ARM = get_age_rating_matrix(URM=URM_train, age_demographic=age_demographic, implicit=False)
    ARM_implicit = get_age_rating_matrix(URM=URM_train, age_demographic=age_demographic, implicit=True)

    # Compute similarity
    similarity_ARM = Compute_Similarity(ARM.T, similarity="cosine")
    similarity_ARM_implicit = Compute_Similarity(ARM_implicit.T, similarity=SIMILARITY_TYPE)

    W_sparse_ARM = similarity_ARM.compute_similarity()
    W_sparse_ARM_implicit = similarity_ARM_implicit.compute_similarity()

    W_sparse_dense_ARM = W_sparse_ARM.todense()
    W_sparse_dense_ARM_implicit = W_sparse_ARM_implicit.todense()

    print(W_sparse_ARM)

    # Plots
    if PLOT_SIMILARITY_MATRIX:
        plt.title("Plot of {} similarity".format(SIMILARITY_TYPE))
        plt.imshow(W_sparse_dense_ARM, interpolation='none', origin="lower")
        plt.colorbar()
        show_fig("w_sparse_ARM")

        plt.title("Plot of {} similarity - Implicit version".format(SIMILARITY_TYPE))
        plt.imshow(W_sparse_dense_ARM_implicit, interpolation='none', origin="lower")
        plt.colorbar()
        show_fig("w_sparse_ARM_implicit")

    if PLOT_SINGLE_GRAPHICS:
        for i, age in enumerate(W_sparse_dense_ARM):
            temp = np.asarray(W_sparse_dense_ARM[i]).squeeze()
            temp = temp[temp != 0]
            plt.title("Age similarity values (Ignore xticks)".format())
            plt.plot(temp)
            show_fig("age_{}_similarity_values".format(i + 1))

        for i, age in enumerate(W_sparse_dense_ARM_implicit):
            temp = np.asarray(W_sparse_dense_ARM_implicit[i]).squeeze()
            temp = temp[temp != 0]
            plt.title("Age {} similarity values - Implicit version (Ignore xticks)".format(i + 1))
            plt.plot(temp)
            show_fig("age_{}_similarity_values_implicit".format(i + 1))
