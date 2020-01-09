from datetime import datetime

import numpy as np

from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import read_target_users, read_URM_cold_all, read_UCM_cold_all, get_ICM_all_new, \
    get_UCM_all_new
from src.model import best_models, best_models_lower_threshold_23, best_models_upper_threshold_22
from src.model.FallbackRecommender.MapperRecommender import MapperRecommender
from src.model.HybridRecommender.HybridDemographicRecommender import HybridDemographicRecommender
from src.model_management.submission_helper import write_submission_file_batch

if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader.load_data()
    URM_all = data_reader.get_URM_all()
    ICM_all = get_ICM_all_new(data_reader)
    UCM_all = get_UCM_all_new(data_reader)

    lt_23_recommender = best_models_lower_threshold_23.WeightedAverageItemBasedWithRP3.get_model(URM_all,
                                                                                                 ICM_all)
    ut_22_recommender = best_models_upper_threshold_22.WeightedAverageAll.get_model(URM_all, ICM_all, UCM_all)
    lt_23_users_mask = np.ediff1d(URM_all.tocsr().indptr) >= 23
    lt_23_users = np.arange(URM_all.shape[0])[lt_23_users_mask]
    ut_23_users = np.arange(URM_all.shape[0])[~lt_23_users_mask]

    main_recommender = HybridDemographicRecommender(URM_train=URM_all)
    main_recommender.add_user_group(1, lt_23_users)
    main_recommender.add_user_group(2, ut_23_users)
    main_recommender.add_relation_recommender_group(lt_23_recommender, 1)
    main_recommender.add_relation_recommender_group(ut_22_recommender, 2)
    main_recommender.fit()

    # Sub recommender
    URM_cold_all = read_URM_cold_all("../data/data_train.csv")
    UCM_cold_all = read_UCM_cold_all(URM_cold_all.shape[0], "../data/")

    sub_recommender = best_models.UserCBF_CF_Cold.get_model(URM_cold_all, UCM_cold_all)

    mapper_model = MapperRecommender(URM_cold_all)
    mapper_model.fit(main_recommender=main_recommender, sub_recommender=sub_recommender,
                     mapper=data_reader.get_user_original_ID_to_index_mapper())
    target_users = read_target_users("../data/data_target_users_test.csv")

    submission_path = "submission_" + datetime.now().strftime('%b%d_%H-%M-%S') + ".csv"
    write_submission_file_batch(mapper_model, submission_path, target_users, batches=20)
