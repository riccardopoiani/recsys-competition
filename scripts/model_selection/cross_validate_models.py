from src.model_management.NewEvaluator import EvaluatorCrossValidationKeepKOut
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from datetime import datetime

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
    results = evaluator.crossevaluateRecommender(ItemKNNCFRecommender, **item_cf_keywargs)

    # Writing on file cross validation results
    f = open(destination_path, "w")
    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n")
    f.write(str(item_cf_keywargs))
    f.write("\n")
    f.write(str(results))
    f.close()