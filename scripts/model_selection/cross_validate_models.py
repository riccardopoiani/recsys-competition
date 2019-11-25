from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.model.HybridRecommender import HybridWeightedAverageRecommender
from src.model_management.NewEvaluator import EvaluatorCrossValidationKeepKOut
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from datetime import datetime


def hybrid_model_get_all_models(URM_train, ICM_numerical, ICM_categorical):
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
    # Set seed in order to have same splitting of data
    seed_list = [1247, 8246, 2346, 1535]

    # Parameters
    item_cf_keywargs = {'topK': 5, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                        'feature_weighting': 'TF-IDF'}
    destination_path = "../../report/cross_validation/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    destination_path = destination_path + "cross_valid_item_cf_" + now +".txt"
    num_folds = len(seed_list)

    # Setting evaluator
    evaluator = EvaluatorCrossValidationKeepKOut(10, seed_list, "../../data/",  n_folds=num_folds)
    results = evaluator.crossevaluateHybridRecommender(HybridWeightedAverageRecommender, item_cf_keywargs)

    # Writing on file cross validation results
    f = open(destination_path, "w")
    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n")
    f.write(str(item_cf_keywargs))
    f.write("\n")
    f.write(str(results))
    f.close()