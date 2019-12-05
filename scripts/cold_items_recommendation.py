import os

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM, get_ICM_numerical
from src.data_management.data_getter import get_warmer_UCM
from src.model import best_models
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Data loading
    data_reader = RecSys2019Reader("../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
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

    cold_items_mask = np.ediff1d(URM_train.tocsc().indptr) < 200
    cold_items = np.arange(URM_train.shape[1])[cold_items_mask]

    warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
    warm_users = np.arange(URM_train.shape[0])[warm_users_mask]

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]

    UCM_age = get_warmer_UCM(UCM_age, URM_all, threshold_users=3)
    UCM_region = get_warmer_UCM(UCM_region, URM_all, threshold_users=3)

    model = best_models.ItemCBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=False)

    recommendations = model.recommend(user_id_array=warm_users, remove_seen_flag=True, cutoff=10,
                                      remove_top_pop_flag=False)

    # Number of cold recommendations on warm users
    count_cold_recommendations = 0

    for i in range(len(recommendations)):
        if i % 10000 == 0:
            print("i {}".format(i))
        count_cold_recommendations += np.isin(np.array(recommendations[i]), cold_items).sum()

    print("Number of cold recommendations {}".format(count_cold_recommendations))

    # Evaluation with cold items
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)
    print("MAP@10 with cold items {}".format(evaluator.evaluateRecommender(model)[0][10]['MAP']))

    # Evaluation preventing the algorithm to recommend cold items
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users,
                                 ignore_items=cold_items)
    print("MAP@10 without cold items {}".format(evaluator.evaluateRecommender(model)[0][10]['MAP']))



