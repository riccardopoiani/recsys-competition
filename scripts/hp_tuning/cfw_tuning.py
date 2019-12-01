from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader, merge_ICM
from src.model.best_models import ItemCF
from src.tuning.run_parameter_search_cfw_linalg import run_parameter_search
from src.utils.general_utility_functions import get_split_seed


if __name__ == '__main__':
    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())

    data_reader.load_data()
    mapper = data_reader.SPLIT_GLOBAL_MAPPER_DICT['user_original_ID_to_index']
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_all, _ = merge_ICM(ICM, URM_train.transpose(), {}, {})

    # Setting evaluator
    cutoff_list = [10]
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    # Path setting
    print("Start tuning...")
    version_path = "../../report/hp_tuning/cf_boosted/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3_eval/"
    version_path = version_path + now

    item_cf = ItemCF.get_model(URM_train, load_saved_model=False)
    W_sparse_CF = item_cf.W_sparse

    # Fit ItemKNN best model and get the sparse matrix of the weights
    run_parameter_search(URM_train=URM_train, output_folder_path=version_path,
                         evaluator_test=evaluator_test,
                         W_sparse_CF=W_sparse_CF, ICM_all=ICM_all)
    print("...tuning ended")
