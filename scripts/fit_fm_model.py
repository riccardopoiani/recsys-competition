import os

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model import new_best_models
from src.model.FactorizationMachine.FactorizationMachineRecommender import FactorizationMachineRecommender
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
    ICM_all = get_ICM_train(data_reader)

    # Build UCMs: do not change the order of ICMs and UCMs
    UCM_all = get_UCM_train(data_reader)

    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
    warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
    ignore_users = np.unique(np.concatenate((cold_users, warm_users)))

    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

    best_models = new_best_models.RP3BetaSideInfo.get_model(URM_train, ICM_all)
    fm_data_path = os.path.join(get_project_root_path(), "resources", "fm_data")
    model = FactorizationMachineRecommender(URM_train,
                                            train_svm_file_path=os.path.join(fm_data_path,
                                                                             "ICM_UCM_uncompressed.txt"),
                                            approximate_recommender=best_models,
                                            ICM_train=ICM_all, UCM_train=UCM_all, max_items_to_predict=20)
    model.fit(latent_factors=100, learning_rate=0.01, epochs=10)
    print(evaluator.evaluateRecommender(model))
