from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model import new_best_models
from src.tuning.holdout_validation.run_parameter_search_cfw_linalg import run_parameter_search
from src.utils.general_utility_functions import get_split_seed


if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())

    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_all = get_ICM_train(data_reader)

    UCM_all = get_UCM_train(data_reader)

    # Setting evaluator
    cutoff_list = [10]
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    # Path setting
    print("Start tuning...")
    version_path = "../../report/hp_tuning/p3alpha_w_sparse/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3_eval/"
    version_path = version_path + now

    user_cf = new_best_models.UserCBF_CF_Warm.get_model(URM_train, UCM_all)
    W_sparse_CF = user_cf.W_sparse

    item_cbf_cf = new_best_models.ItemCBF_CF.get_model(URM_train, ICM_all)

    # Fit ItemKNN best model and get the sparse matrix of the weights
    run_parameter_search(URM_train=URM_train, output_folder_path=version_path,
                         evaluator_test=evaluator_test,
                         W_sparse_CF=item_cbf_cf.W_sparse, ICM_all=W_sparse_CF, n_cases=60, n_random_starts=20)
    print("...tuning ended")
