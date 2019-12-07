import os

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Base.NonPersonalizedRecommender import TopPop
from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.data_management.DataPreprocessing import DataPreprocessingDigitizeICMs
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM, get_ICM_numerical, get_UCM_all
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.model.FactorMachines.FactorizationMachineRecommender import FactorizationMachineRecommender
from src.utils.general_utility_functions import get_split_seed, get_project_root_path

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Data loading
    data_reader = RecSys2019Reader("../data/")
    data_reader = DataPreprocessingDigitizeICMs(data_reader, ICM_name_to_bins_mapper={"ICM_asset": 50, "ICM_price": 50,
                                                                                      "ICM_item_pop": 20})
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM = data_reader.get_ICM_from_name("ICM_all")
    ICM_subclass = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_all, _ = merge_ICM(ICM, URM_train.transpose(), {}, {})
    ICM_subclass_all, _ = merge_ICM(ICM_subclass, URM_train.transpose(), {}, {})

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_all = get_UCM_all(data_reader.dataReader_object.reader, discretize_user_act_bins=20)
    UCM_train = get_warmer_UCM(UCM_all, URM_all, threshold_users=3)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    sub = best_models.ItemCBF_CF.get_model(URM_train, ICM_all)

    ICM_all_digitized = data_reader.get_ICM_from_name("ICM_all")
    root_path = get_project_root_path()
    train_svm_file_path = os.path.join(root_path, "resources", "fm_data", "URM_ICM_UCM_uncompressed.txt")
    model = FactorizationMachineRecommender(URM_train, train_svm_file_path, sub, ICM_train=ICM_all_digitized,
                                            UCM_train=UCM_train)
    model.fit(epochs=100, latent_factors=100, regularization=10e-8, learning_rate=0.1)

    print(evaluator.evaluateRecommender(model))
