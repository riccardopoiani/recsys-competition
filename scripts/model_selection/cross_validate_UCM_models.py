from datetime import datetime

from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.model_management.CrossEvaluator import EvaluatorCrossValidationKeepKOut

if __name__ == '__main__':
    # Set seed in order to have same splitting of data
    seed_list = [1247, 8246, 2346, 1535, 3455]

    # Parameters
    user_cbf_kwargs = {'topK': 3285, 'shrink': 1189, 'similarity': 'cosine',
                       'normalize': False, 'feature_weighting': 'BM25'}

    destination_path = "../../report/cross_validation/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    destination_path = destination_path + "cross_valid_user_cbf_" + now +".txt"
    num_folds = len(seed_list)

    data_reader = RecSys2019Reader("../../data")
    data_reader.load_data()
    URM_all = data_reader.get_URM_all()

    # Setting evaluator
    evaluator = EvaluatorCrossValidationKeepKOut(10, seed_list, "../../data/",  n_folds=num_folds)
    results = evaluator.crossevaluateUCMRecommender(UserKNNCBFRecommender, on_cold_users=True, **user_cbf_kwargs)

    # Writing on file cross validation results
    f = open(destination_path, "w")
    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n")
    f.write(str(user_cbf_kwargs))
    f.write("\n")
    f.write(str(results))
    f.close()