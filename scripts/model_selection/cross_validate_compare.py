from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.DataPreprocessing import DataPreprocessingFeatureEngineering, DataPreprocessingImputation, \
    DataPreprocessingTransform, DataPreprocessingDiscretization
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical
from src.data_management.data_getter import get_UCM_train
from src.model import new_best_models
from src.utils.general_utility_functions import get_split_seed
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender

if __name__ == '__main__':
    seed_list = [6910, 1996, 2019, 153, 12, 5, 1010, 9999, 666, 467, 6, 151, 86, 99, 4444]

    new_best_param = {'topK': 7, 'shrink': 1494, 'similarity': 'cosine', 'normalize': True,
                      'feature_weighting': 'TF-IDF', 'interactions_feature_weighting': 'TF-IDF'}
    model_score = 0

    for i in range(0, len(seed_list)):
        # Data loading
        root_data_path = "../data/"
        data_reader = RecSys2019Reader(root_data_path)
        data_reader = DataPreprocessingFeatureEngineering(data_reader,
                                                          ICM_names_to_count=["ICM_sub_class"])
        data_reader = DataPreprocessingImputation(data_reader,
                                                  ICM_name_to_agg_mapper={"ICM_asset": np.median,
                                                                          "ICM_price": np.median})
        data_reader = DataPreprocessingTransform(data_reader,
                                                 ICM_name_to_transform_mapper={"ICM_asset": lambda x: np.log1p(1 / x),
                                                                               "ICM_price": lambda x: np.log1p(1 / x),
                                                                               "ICM_item_pop": np.log1p,
                                                                               "ICM_sub_class_count": np.log1p})
        data_reader = DataPreprocessingDiscretization(data_reader,
                                                      ICM_name_to_bins_mapper={"ICM_asset": 200,
                                                                               "ICM_price": 200,
                                                                               "ICM_item_pop": 50,
                                                                               "ICM_sub_class_count": 50})
        data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=False,
                                                   force_new_split=True, seed=get_split_seed())
        data_reader.load_data()
        URM_train, URM_test = data_reader.get_holdout_split()

        # Build ICMs
        ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
        ICM_all = data_reader.get_ICM_from_name("ICM_all")

        # Build UCMs: do not change the order of ICMs and UCMs
        UCM_all = get_UCM_train(data_reader, root_data_path)

        # Setting evaluator
        cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
        cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
        warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) < 80
        warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
        ignore_users = np.unique(np.concatenate((cold_users, warm_users)))

        cutoff_list = [10]
        evaluator_total = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)

        # model = new_best_models.ItemCBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all)
        #model = ItemKNNCBFCFRecommender(URM_train=URM_train, ICM_train=ICM_all)
        #model.fit(**new_best_param)
        model = new_best_models.ItemCBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all)

        curr_map = evaluator_total.evaluateRecommender(model)[0][10]['MAP']

        print("SEED: {} \n ".format(seed_list[i]))
        print("CURR MAP {} \n ".format(curr_map))

        model_score += curr_map

    model_score /= len(seed_list)

    # Store results on file
    destination_path = "../../report/cross_validation/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    destination_path = destination_path + "cross_valid_comparison" + now + ".txt"
    num_folds = len(seed_list)

    f = open(destination_path, "w")
    f.write("X-validation ItemCBFCF new best model FOH 80\n\n")

    f.write("X-val map {} \n\n".format(model_score))

    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n\n")

    f.close()
