from datetime import datetime

from course_lib.Base.NonPersonalizedRecommender import TopPop
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import read_target_users, read_URM_cold_all, read_UCM_cold_all, get_UCM_all, \
    get_ICM_all
from src.model import best_models, new_best_models
from src.model.FallbackRecommender.MapperRecommender import MapperRecommender
from src.model.HybridRecommender.HybridDemographicRecommender import HybridDemographicRecommender
from src.model_management.submission_helper import write_submission_file_batch

import os
import pandas as pd
import scipy.sparse as sps
import numpy as np

if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader.load_data()
    URM_all = data_reader.get_URM_all()
    ICM_all = get_ICM_all(data_reader)
    UCM_all = get_UCM_all(data_reader)

    # Main recommender
    main_recommender = new_best_models.MixedItem.get_model(URM_all, ICM_all)

    # Sub recommender
    URM_cold_all = read_URM_cold_all("../data/data_train.csv")
    UCM_cold_all = read_UCM_cold_all(URM_cold_all.shape[0], "../data/")

    df_age = pd.read_csv(os.path.join("../data/", "data_UCM_age.csv"))

    user_id_list = df_age['row'].values
    age_id_list = df_age['col'].values
    UCM_age = sps.coo_matrix((np.ones(len(user_id_list)), (user_id_list, age_id_list)),
                             shape=(UCM_cold_all.shape[0], np.max(age_id_list) + 1))
    cold_users_age_mask = np.ediff1d(UCM_age.tocsr().indptr) == 0
    cold_users_age = np.arange(UCM_age.shape[0])[cold_users_age_mask]
    warm_users_age = np.arange(UCM_age.shape[0])[~cold_users_age_mask]
    print(len(cold_users_age))

    """topPop = TopPop(URM_cold_all)
    topPop.fit()
    user_cbf_cf = best_models.UserCBF_CF_Cold.get_model(URM_cold_all, UCM_cold_all)
    sub_recommender = HybridDemographicRecommender(URM_cold_all)
    sub_recommender.add_user_group(0, warm_users_age)
    sub_recommender.add_user_group(1, cold_users_age)
    sub_recommender.add_relation_recommender_group(user_cbf_cf, 0)
    sub_recommender.add_relation_recommender_group(topPop, 1)
    sub_recommender.fit()"""
    sub_recommender = best_models.UserCBF_CF_Cold.get_model(URM_cold_all, UCM_cold_all)

    mapper_model = MapperRecommender(URM_cold_all)
    mapper_model.fit(main_recommender=main_recommender, sub_recommender=sub_recommender,
                     mapper=data_reader.get_user_original_ID_to_index_mapper())
    target_users = read_target_users("../data/data_target_users_test.csv")

    submission_path = "submission_" + datetime.now().strftime('%b%d_%H-%M-%S') + ".csv"
    write_submission_file_batch(mapper_model, submission_path, target_users, batches=20)
