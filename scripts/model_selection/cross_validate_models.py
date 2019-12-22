from datetime import datetime

from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from src.model import best_models, new_best_models
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.model_management.CrossEvaluator import EvaluatorCrossValidationKeepKOut
from src.utils.general_utility_functions import get_seed_lists, get_split_seed

if __name__ == '__main__':
    # Set seed in order to have same splitting of data
    seed_list = [6910, 1996, 2019, 153, 12, 5, 1010, 9999, 666, 467]

    # Parameters
    destination_path = "../../report/cross_validation/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    destination_path = destination_path + "cross_valid_item_cf_" + now +".txt"
    num_folds = len(seed_list)

    model_parameters = {'topK': 12, 'shrink': 1656, 'normalize': True, 'interactions_feature_weighting': 'TF-IDF',
           'similarity': 'asymmetric', 'asymmetric_alpha': 0.029336346124356323, 'feature_weighting': 'TF-IDF'}

    # Setting evaluator
    evaluator = EvaluatorCrossValidationKeepKOut(10, seed_list, "../../data/",  k_out=1, n_folds=num_folds)
    results = evaluator.crossevaluateCBFRecommender(ItemKNNCBFCFRecommender, **model_parameters)

    # Writing on file cross validation results
    f = open(destination_path, "w")
    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n")
    f.write("\n")
    f.write(str(results))
    f.close()