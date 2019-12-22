from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.DataPreprocessing import DataPreprocessingRemoveColdUsersItems
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train, get_ICM_train_new
from src.model import new_best_models
from src.tuning.holdout_validation.run_parameter_search_cfw_linalg import run_parameter_search
from src.utils.general_utility_functions import get_split_seed
import scipy.sparse as sps


if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = DataPreprocessingRemoveColdUsersItems(data_reader, threshold_items=-1, threshold_users=20)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=True,
                                               force_new_split=True, seed=get_split_seed())

    data_reader.load_data()
    URM_train, URM_valid, URM_test = data_reader.get_holdout_split()
    ICM_all, _ = get_ICM_train_new(data_reader)

    #UCM_all = get_UCM_train(data_reader)

    # Setting evaluator
    cutoff_list = [10]
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    # Path setting
    print("Start tuning...")
    version_path = "../../report/hp_tuning/p3alpha_w_sparse/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3_eval/"
    version_path = version_path + now

    URM_train_valid = URM_train + URM_valid
    item_cbf_cf = new_best_models.ItemCBF_CF.get_model(URM_train_valid, ICM_all)

    # Fit ItemKNN best model and get the sparse matrix of the weights
    run_parameter_search(URM_train=URM_train, output_folder_path=version_path,
                         evaluator_test=evaluator_test,
                         W_sparse_CF=item_cbf_cf.W_sparse, ICM_all=sps.hstack([URM_train.T, ICM_all]),
                         n_cases=100, n_random_starts=40)
    print("...tuning ended")
