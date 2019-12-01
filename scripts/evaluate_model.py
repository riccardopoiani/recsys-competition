from datetime import datetime

from lightfm.evaluation import precision_at_k
from numpy.random import seed

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM, get_ICM_numerical
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.model.KNN.UserItemCBFCFDemographicRecommender import UserItemCBFCFDemographicRecommender
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from src.tuning.run_parameter_search_user_item_all import run_parameter_search_user_item_all
from src.data_management.DataPreprocessing import DataPreprocessingRemoveColdUsersItems
from src.model.MatrixFactorization.LightFMRecommender import LightFMRecommender

SEED = 69420

if __name__ == '__main__':
    # Set seed in order to have same splitting of data
    seed(SEED)

    # Data loading
    data_reader = RecSys2019Reader("../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_all, _ = merge_ICM(ICM, URM_train.transpose(), {}, {})

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
    UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})

    UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)
    UCM_all, _ = merge_UCM(UCM_age_region, URM_train, {}, {})

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]

    cold_items_mask = np.ediff1d(URM_train.tocsc().indptr) == 0
    cold_items = np.arange(URM_train.shape[1])[cold_items_mask]

    # Reset seed for hyper-parameter tuning
    seed()

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    model = ImplicitALSRecommender(URM_train)
    model.fit(**{'num_factors': 488, 'regularization': 93.82377291481401, 'epochs': 50,
               'confidence_scaling': 'log', 'alpha': 1.6388712706938415, 'epsilon': 0.0012718884782782912})
    print(evaluator.evaluateRecommender(model))

