from datetime import datetime

from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM
from src.data_management.data_reader import read_target_users, read_URM_cold_all, read_UCM_cold_all
from src.model import best_models
from src.model.FallbackRecommender.MapperRecommender import MapperRecommender
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.model_management.submission_helper import write_submission_file_batch

if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader.load_data()
    URM_all = data_reader.get_URM_all()
    ICM_categorical = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_all, _ = merge_ICM(ICM_categorical, URM_all.T, {}, {})

    UCM_age = data_reader.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.get_UCM_from_name("UCM_region")
    UCM_all, _ = merge_UCM(UCM_age, UCM_region, {}, {})
    UCM_all, _ = merge_UCM(UCM_all, URM_all, {}, {})

    # Main recommender
    main_recommender = best_models.WeightedAverageMixed.get_model(URM_train=URM_all, ICM_all=ICM_all, UCM_all=UCM_all,
                                                                  ICM_subclass_all=ICM_all)

    # Sub recommender
    URM_cold_all = read_URM_cold_all("../data/data_train.csv")
    UCM_cold_all = read_UCM_cold_all(URM_cold_all.shape[0], "../data/")
    UCM_cold_all, _ = merge_UCM(UCM_cold_all, URM_cold_all, {}, {})
    sub_recommender = UserKNNCBFRecommender(URM_cold_all, UCM_cold_all)
    sub_recommender.fit()

    mapper_model = MapperRecommender(URM_cold_all)
    mapper_model.fit(main_recommender=main_recommender, sub_recommender=sub_recommender,
                     mapper=data_reader.get_user_original_ID_to_index_mapper())
    target_users = read_target_users("../data/data_target_users_test.csv")

    submission_path = "submission_" + datetime.now().strftime('%b%d_%H-%M-%S') + ".csv"
    write_submission_file_batch(mapper_model, submission_path, target_users, batches=20)
