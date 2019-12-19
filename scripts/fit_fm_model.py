import os

from course_lib.Base.Evaluation.Evaluator import *
from course_lib.Base.NonPersonalizedRecommender import TopPop
from scripts.fm_model.write_ffm_data_uncompressed import get_ICM_with_fields, get_UCM_with_fields
from src.data_management.DataPreprocessing import DataPreprocessingRemoveColdUsersItems
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.model import new_best_models
from src.model.FactorizationMachine.FieldAwareFMRecommender import FieldAwareFMRecommender
from src.utils.general_utility_functions import get_split_seed, get_project_root_path

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Data loading
    root_data_path = "../data/"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all, item_feature_fields = get_ICM_with_fields(data_reader)

    # Build UCMs: do not change the order of ICMs and UCMs
    UCM_all, user_feature_fields = get_UCM_with_fields(data_reader)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    best_model = new_best_models.ItemCBF_CF.get_model(URM_train, ICM_all)
    best_model.fit()
    ffm_data_path = os.path.join(get_project_root_path(), "resources", "ffm_data")
    model = FieldAwareFMRecommender(URM_train, model_type="ffm",
                                    train_svm_file_path=os.path.join(ffm_data_path, "train_uncompressed.txt"),
                                    valid_svm_file_path=os.path.join(ffm_data_path, "valid_uncompressed.txt"),
                                    approximate_recommender=best_model, ICM_train=ICM_all, UCM_train=UCM_all,
                                    item_feature_fields=item_feature_fields, user_feature_fields=user_feature_fields,
                                    max_items_to_predict=100)
    #model.load_model(os.path.join(ffm_data_path, "model"), "model_row_4.out")
    model.fit(latent_factors=100, learning_rate=0.01, epochs=100, regularization=1e-2, stop_window=4)
    print(evaluator.evaluateRecommender(model))
