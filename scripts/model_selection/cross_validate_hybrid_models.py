from datetime import datetime

from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from src.model import best_models
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from src.model_management.CrossEvaluator import EvaluatorCrossValidationKeepKOut


def hybrid_model_get_all_models():
    all_models = {}
    all_models_fit_parameters = {}
    all_models_constructor_parameters = {}

    all_models['ITEM_CF'] = ItemKNNCFRecommender
    all_models_fit_parameters['ITEM_CF'] = best_models.ItemCF.get_best_parameters()
    all_models_constructor_parameters['ITEM_CF'] = {}

    all_models['USER_CF'] = UserKNNCFRecommender
    all_models_fit_parameters['USER_CF'] = best_models.UserCF.get_best_parameters()
    all_models_constructor_parameters['USER_CF'] = {}

    return all_models, all_models_fit_parameters, all_models_constructor_parameters


if __name__ == '__main__':
    # Set seed in order to have same splitting of data
    seed_list = [1247, 8246, 2346, 1535]

    # Parameters
    hybrid_kwargs = {'MIXED_ITEM': 0.014667586445465623, 'MIXED_USER': 0.0013235051989859417}
    destination_path = "../../report/cross_validation/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    destination_path = destination_path + "cross_valid_item_cf_" + now +".txt"
    num_folds = len(seed_list)

    all_models, all_models_fit, all_models_constructor = hybrid_model_get_all_models()

    # Setting evaluator
    evaluator = EvaluatorCrossValidationKeepKOut(10, seed_list, "../../data/",  n_folds=num_folds)
    results = evaluator.crossevaluateHybridRecommender(HybridWeightedAverageRecommender, hybrid_kwargs, all_models,
                                                       all_models_constructor, all_models_fit)

    # Writing on file cross validation results
    f = open(destination_path, "w")
    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n")
    f.write(str(hybrid_kwargs))
    f.write("\n")
    f.write(str(results))
    f.close()