import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from course_lib.Base.Similarity.Compute_Similarity import Compute_Similarity
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.feature.demographics_content import get_sub_class_content
from src.model.NewRatingMatrix import get_subclass_rating_matrix
from src.utils.general_utility_functions import get_split_seed

# SETTINGS
SIMILARITY_TYPE = "cosine"
KEEP_OUT = 1
SAVE_ON_FILE = True
PLOT_SIMILARITY_MATRIX = True
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
        version_path = "../../report/graphics/SRM/{}/".format(SIMILARITY_TYPE)
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

    ICM_subclass = data_reader.get_ICM_from_name("ICM_sub_class")
    subclass_feature_to_id_mapper = data_reader.dataReader_object.get_ICM_feature_to_index_mapper_from_name(
        "ICM_sub_class")
    subclass_content_dict = get_sub_class_content(ICM_subclass, subclass_feature_to_id_mapper, binned=True)
    subclass_content = get_sub_class_content(ICM_subclass, subclass_feature_to_id_mapper, binned=False)

    SRM = get_subclass_rating_matrix(URM=URM_train, subclass_content_dict=subclass_content_dict)
    SRM_implicit = get_subclass_rating_matrix(URM=URM_train, subclass_content_dict=subclass_content_dict, implicit=True)

    # Compute similarity
    similarity_SRM = Compute_Similarity(SRM, similarity="cosine")
    similarity_SRM_implicit = Compute_Similarity(SRM_implicit, similarity=SIMILARITY_TYPE)

    W_sparse_SRM = similarity_SRM.compute_similarity()
    W_sparse_SRM_implicit = similarity_SRM_implicit.compute_similarity()

    W_sparse_dense_SRM = W_sparse_SRM.todense()
    W_sparse_dense_SRM_implicit = W_sparse_SRM_implicit.todense()

    print(W_sparse_SRM)

    # Plots
    if PLOT_SIMILARITY_MATRIX:
        plt.title("Plot of {} similarity".format(SIMILARITY_TYPE))
        plt.imshow(W_sparse_dense_SRM, interpolation='none', origin="lower")
        plt.colorbar()
        show_fig("w_sparse_SRM")

        plt.title("Plot of {} similarity - Implicit version".format(SIMILARITY_TYPE))
        plt.imshow(W_sparse_dense_SRM_implicit, interpolation='none', origin="lower")
        plt.colorbar()
        show_fig("w_sparse_SRM_implicit")

    if PLOT_SINGLE_GRAPHICS:
        for i, age in enumerate(W_sparse_dense_SRM):
            temp = np.asarray(W_sparse_dense_SRM[i]).squeeze()
            temp = temp[temp != 0]
            plt.title("Age similarity values (Ignore xticks)".format())
            plt.plot(temp)
            show_fig("age_{}_similarity_values".format(i + 1))

        for i, age in enumerate(W_sparse_dense_SRM_implicit):
            temp = np.asarray(W_sparse_dense_SRM_implicit[i]).squeeze()
            temp = temp[temp != 0]
            plt.title("Age {} similarity values - Implicit version (Ignore xticks)".format(i + 1))
            plt.plot(temp)
            show_fig("age_{}_similarity_values_implicit".format(i + 1))
