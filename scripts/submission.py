from datetime import datetime

from course_lib.Base.NonPersonalizedRecommender import TopPop
from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical
from src.data_management.data_reader import read_target_users, read_URM_cold_all
from src.model.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from src.model.MapperRecommender import MapperRecommender
from src.model_management.submission_helper import write_submission_file_all


def _get_all_models(URM_train, ICM_numerical, ICM_categorical):
    all_models = {}

    item_cf_keywargs = {'topK': 5, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                        'feature_weighting': 'TF-IDF'}
    item_cf = ItemKNNCFRecommender(URM_train)
    item_cf.fit(**item_cf_keywargs)
    all_models['ITEM_CF'] = item_cf

    user_cf_keywargs = {'topK': 995, 'shrink': 9, 'similarity': 'cosine', 'normalize': True,
                        'feature_weighting': 'TF-IDF'}
    user_cf = UserKNNCFRecommender(URM_train)
    user_cf.fit(**user_cf_keywargs)
    all_models['USER_CF'] = user_cf

    item_cbf_numerical_kwargs = {'feature_weighting': 'none', 'normalize': False, 'normalize_avg_row': True,
                                 'shrink': 0, 'similarity': 'euclidean', 'similarity_from_distance_mode': 'exp',
                                 'topK': 1000}
    item_cbf_numerical = ItemKNNCBFRecommender(ICM_numerical, URM_train)
    item_cbf_numerical.fit(**item_cbf_numerical_kwargs)
    item_cbf_numerical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderNumerical"
    all_models['ITEM_CBF_NUM'] = item_cbf_numerical

    item_cbf_categorical_kwargs = {'topK': 5, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                                   'asymmetric_alpha': 2.0, 'feature_weighting': 'BM25'}
    item_cbf_categorical = ItemKNNCBFRecommender(ICM_categorical, URM_train)
    item_cbf_categorical.fit(**item_cbf_categorical_kwargs)
    item_cbf_categorical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderCategorical"
    all_models['ITEM_CBF_CAT'] = item_cbf_categorical

    slim_bpr_kwargs = {'topK': 5, 'epochs': 1499, 'symmetric': False, 'sgd_mode': 'adagrad',
                       'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001}
    slim_bpr = SLIM_BPR_Cython(URM_train)
    slim_bpr.fit(**slim_bpr_kwargs)
    all_models['SLIM_BPR'] = slim_bpr

    p3alpha_kwargs = {'topK': 84, 'alpha': 0.6033770403001427, 'normalize_similarity': True}
    p3alpha = P3alphaRecommender(URM_train)
    p3alpha.fit(**p3alpha_kwargs)
    all_models['P3ALPHA'] = p3alpha

    rp3beta_kwargs = {'topK': 5, 'alpha': 0.37829128706576887, 'beta': 0.0, 'normalize_similarity': False}
    rp3beta = RP3betaRecommender(URM_train)
    rp3beta.fit(**rp3beta_kwargs)
    all_models['RP3BETA'] = rp3beta

    return all_models


if __name__ == '__main__':
    data_reader = RecSys2019Reader("../../data/")
    data_reader.load_data()
    URM_all = data_reader.get_URM_all()
    ICM_categorical = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_numerical, _ = get_ICM_numerical(data_reader)

    # Main recommender
    main_recommender = HybridWeightedAverageRecommender(URM_all)
    all_models = _get_all_models(URM_train=URM_all, ICM_numerical=ICM_numerical,
                                 ICM_categorical=ICM_categorical)
    for model_name, model_object in all_models.items():
        main_recommender.add_fitted_model(model_name, model_object)
    print("The models added in the hybrid are: {}".format(list(all_models.keys())))
    weights = {'ITEM_CF': 0.969586046573504, 'USER_CF': 0.943330450168123, 'ITEM_CBF_NUM': 0.03250599212747674,
               'ITEM_CBF_CAT': 0.018678076600871066, 'SLIM_BPR': 0.03591603993769955,
               'P3ALPHA': 0.7474845972085382, 'RP3BETA': 0.1234024366177027}
    main_recommender.fit(**weights)
    print(main_recommender.weights)

    # Sub recommender
    URM_cold_all = read_URM_cold_all("../data/data_train.csv")
    sub_recommender = TopPop(URM_cold_all)
    sub_recommender.fit()

    mapper_model = MapperRecommender(URM_cold_all)
    mapper_model.fit(main_recommender=main_recommender, sub_recommender=sub_recommender,
                     mapper=data_reader.get_user_original_ID_to_index_mapper())
    target_users = read_target_users("../data/data_target_users_test.csv")

    submission_path = "submission_" + datetime.now().strftime('%b%d_%H-%M-%S') + ".csv"
    write_submission_file_all(mapper_model, submission_path, target_users)
