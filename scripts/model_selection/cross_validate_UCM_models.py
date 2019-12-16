from datetime import datetime

from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.model_management.CrossEvaluator import EvaluatorCrossValidationKeepKOut

if __name__ == '__main__':
    # Set seed in order to have same splitting of data
    seed_list = [6910, 1996, 2019, 153, 12, 5, 1010, 9999, 666, 467, 241, 4124, 5131, 1214, 5123]

    # Parameters
    user_cbf_kwargs = {'topK': 2973, 'shrink': 117, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.007315425738737337, 'feature_weighting': 'BM25',
                       'interactions_feature_weighting': 'TF-IDF'}

    destination_path = "../../report/cross_validation/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    destination_path = destination_path + "cross_valid_user_cbf_" + now +".txt"
    num_folds = len(seed_list)

    data_reader = RecSys2019Reader("../../data")
    data_reader.load_data()
    URM_all = data_reader.get_URM_all()

    # Setting evaluator
    evaluator = EvaluatorCrossValidationKeepKOut(10, seed_list, "../../data/", k_out=1, n_folds=num_folds)
    results = evaluator.crossevaluateDemographicRecommender(UserKNNCBFCFRecommender, on_cold_users=True,
                                                            **user_cbf_kwargs)

    # Writing on file cross validation results
    f = open(destination_path, "w")
    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n")
    f.write(str(user_cbf_kwargs))
    f.write("\n")
    f.write(str(results))
    f.close()