import matplotlib.pyplot as plt
from datetime import datetime
import os

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_UCM_train, get_ICM_train_new
from src.model import new_best_models
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.utils.general_utility_functions import get_split_seed

TOP_K_LIST = [5, 10, 17, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
FOH_THRESHOLDS = [25, 50, 75]
K_OUT = 1
SAVE_ON_FILE = True


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
        version_path = "../../report/graphics/exploration/topk_map/"
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

    root_data_path = "../../data"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=K_OUT, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    ICM_all, _ = get_ICM_train_new(data_reader)
    UCM_all = get_UCM_train(data_reader)

    item_cbf_cf_best_parameters = new_best_models.ItemCBF_CF.get_best_parameters()

    # Building path
    version_path = "../../report/graphics/comparison/topk/"

    # Set up evaluators
    evaluator_list = []
    total_users = np.arange(URM_train.shape[0])
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = total_users[cold_users_mask]
    evaluator = EvaluatorHoldout(URM_test, [10], ignore_users=cold_users)
    evaluator_list.append(evaluator)

    for foh in FOH_THRESHOLDS:
        ignore_users = np.ediff1d(URM_train.tocsr().indptr) < foh
        ignore_users = total_users[ignore_users]
        evaluator_list.append(EvaluatorHoldout(URM_test, [10], ignore_users=ignore_users))

    y_ticks = []
    for i in range(len(evaluator_list)):
        y_ticks.append([])

    for topk in TOP_K_LIST:
        model = ItemKNNCBFCFRecommender(URM_train=URM_train, ICM_train=ICM_all)
        item_cbf_cf_best_parameters["topK"] = topk
        model.fit(**item_cbf_cf_best_parameters)
        for i, eval in enumerate(evaluator_list):
            y_ticks[i].append(eval.evaluateRecommender(model)[0][10]['MAP'])

    plt.title("Top k Item CBF CF - New best models - All users")
    plt.xlabel("Top k")
    plt.ylabel("MAP@10 - All users")
    plt.plot(y_ticks[0])
    show_fig("map_all_users")

    for i, eval in enumerate(evaluator_list):
        if i != 0:
            plt.title("Top k Item CBF CF - New best models - Foh {}".format(FOH_THRESHOLDS[i-1]))
            plt.xlabel("Top k")
            plt.ylabel("MAP@10 - All users")
            plt.plot(y_ticks[i])
            show_fig("map_foh_{}".format(FOH_THRESHOLDS[i-1]))
