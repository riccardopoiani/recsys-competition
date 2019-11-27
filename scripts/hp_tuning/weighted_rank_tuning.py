from datetime import datetime

from numpy.random import seed

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Base.NonPersonalizedRecommender import TopPop
from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical
from src.model.HybridRecommender.HybridRankBasedRecommender import HybridRankBasedRecommender
from src.tuning.run_parameter_search_hybrid import run_parameter_search_hybrid

SEED = 69420


def _get_all_models(URM_train, ICM_numerical, ICM_categorical):
    all_models = {}

    topPop = TopPop(URM_train)
    topPop.fit()

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

    slim_bpr_kwargs = {'topK': 5, 'epochs': 1499, 'symmetric': False, 'sgd_mode': 'adagrad',
                       'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001}
    slim_bpr = SLIM_BPR_Cython(URM_train)
    slim_bpr.fit(**slim_bpr_kwargs)
    all_models['SLIM_BPR'] = slim_bpr

    item_cbf_numerical_kwargs = {'feature_weighting': 'none', 'normalize': False, 'normalize_avg_row': True,
                                 'shrink': 0, 'similarity': 'euclidean', 'similarity_from_distance_mode': 'exp',
                                 'topK': 1000}
    item_cbf_numerical = ItemKNNCBFRecommender(URM_train, ICM_numerical)
    item_cbf_numerical.fit(**item_cbf_numerical_kwargs)
    item_cbf_numerical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderNumerical"
    all_models['ITEM_CBF_NUM'] = item_cbf_numerical

    item_cbf_categorical_kwargs = {'topK': 5, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                                   'asymmetric_alpha': 2.0, 'feature_weighting': 'BM25'}
    item_cbf_categorical = ItemKNNCBFRecommender(URM_train, ICM_categorical)
    item_cbf_categorical.fit(**item_cbf_categorical_kwargs)
    item_cbf_categorical.RECOMMENDER_NAME = "ItemCBFKNNRecommenderCategorical"
    all_models['ITEM_CBF_CAT'] = item_cbf_categorical

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
    # Set seed in order to have same splitting of data
    seed(SEED)

    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_categorical = data_reader.get_ICM_from_name("ICM_sub_class")
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)

    # Reset seed for hyper-parameter tuning
    seed()

    model = HybridRankBasedRecommender(URM_train)

    all_models = _get_all_models(URM_train=URM_train, ICM_numerical=ICM_numerical,
                                 ICM_categorical=ICM_categorical)
    for model_name, model_object in all_models.items():
        model.add_fitted_model(model_name, model_object)
    print("The models added in the hybrid are: {}".format(list(all_models.keys())))

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    version_path = "../../report/hp_tuning/hybrid_weighted_rank/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + "/" + now

    run_parameter_search_hybrid(model, metric_to_optimize="MAP",
                                evaluator_validation=evaluator,
                                output_folder_path=version_path,
                                n_cases=35, parallelizeKNN=False)

    print("...tuning ended")
