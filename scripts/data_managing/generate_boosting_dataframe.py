import os

from course_lib.Base.NonPersonalizedRecommender import TopPop
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_UCM_train, get_ignore_users, get_ICM_train_new
from src.model import new_best_models, best_models
from src.model.Ensemble.Boosting.boosting_preprocessing import get_train_dataframe_proportion, \
    get_valid_dataframe_second_version, get_dataframe_all_data
from src.model.MatrixFactorization.NewPureSVDRecommender import NewPureSVDRecommender
from src.utils.general_utility_functions import get_split_seed

LOWER_THRESHOLD = 20
VALID_CUTOFF = 100
TRAIN_CUTOFF = 100
NEGATIVE_LABEL_VALUE = 0
IGNORE_NON_TARGET_USERS = True
NEGATIVE_PROPORTION = 1

if __name__ == '__main__':
    # Data loading
    root_data_path = "../../data/"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all, _ = get_ICM_train_new(data_reader)

    # Build UCMs: do not change the order of ICMs and UCMs
    UCM_all = get_UCM_train(data_reader)

    # XGB SETUP
    main_rec = new_best_models.MixedItem.get_model(URM_train=URM_train, ICM_all=ICM_all, load_model=True,
                                                   save_model=True)
    sub_0 = main_rec
    sub_0.RECOMMENDER_NAME = "MixedItem"
    sub_1 = new_best_models.ItemCBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=True,
                                                 save_model=True)
    sub_1.RECOMMENDER_NAME = "ItemCBF_CF"
    sub_2 = new_best_models.RP3BetaSideInfo.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=True,
                                                      save_model=True)
    sub_2.RECOMMENDER_NAME = "RP3BetaSideInfo"
    sub_3 = new_best_models.UserCBF_CF_Warm.get_model(URM_train=URM_train, UCM_train=UCM_all, load_model=True,
                                                      save_model=True)
    sub_3.RECOMMENDER_NAME = "UserCBF_CF"
    sub_4 = new_best_models.ItemCBF_all_FW.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=True,
                                                     save_model=True)
    sub_4.RECOMMENDER_NAME = "ItemCBF_all_FW"
    sub_5 = new_best_models.UserCF.get_model(URM_train=URM_train, load_model=True, save_model=True)
    sub_5.RECOMMENDER_NAME = "UserCF"

    sub_6 = best_models.ItemCF.get_model(URM_train=URM_train, load_model=True, save_model=True)
    sub_6.RECOMMENDER_NAME = "ItemCF"

    sub_7 = best_models.SLIM_BPR.get_model(URM_train=URM_train, load_model=True, save_model=True)

    sub_list = [sub_0, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7]

    pure_svd_param = {'num_factors': 50, 'n_oversamples': 3, 'n_iter': 20, 'feature_weighting': 'TF-IDF'}
    pure_svd = NewPureSVDRecommender(URM_train)
    pure_svd.fit(**pure_svd_param)
    user_factors = np.array(pure_svd.USER_factors)
    item_factors = np.array(pure_svd.ITEM_factors)

    mapper = data_reader.get_original_user_id_to_index_mapper()
    ignore_users = get_ignore_users(URM_train, mapper, lower_threshold=LOWER_THRESHOLD, upper_threshold=2 ** 16 - 1,
                                    ignore_non_target_users=IGNORE_NON_TARGET_USERS)
    main_recommender = main_rec
    total_users = np.arange(URM_train.shape[0])
    mask = np.in1d(total_users, ignore_users, invert=True)
    user_to_validate = total_users[mask]
    data_path = "../../data/"

    # Retrieve data for boosting

    train_df = get_train_dataframe_proportion(user_id_array=user_to_validate,
                                              cutoff=TRAIN_CUTOFF,
                                              main_recommender=main_recommender,
                                              recommender_list=sub_list,
                                              mapper=mapper,
                                              URM_train=URM_train,
                                              user_factors=user_factors,
                                              item_factors=item_factors,
                                              path=data_path, negative_label_value=NEGATIVE_LABEL_VALUE,
                                              proportion=NEGATIVE_PROPORTION,
                                              threshold=0.3)

    valid_df = get_valid_dataframe_second_version(user_id_array=user_to_validate,
                                                  cutoff=VALID_CUTOFF,
                                                  main_recommender=main_recommender,
                                                  recommender_list=sub_list,
                                                  mapper=mapper,
                                                  URM_train=URM_train,
                                                  user_factors=user_factors,
                                                  item_factors=item_factors,
                                                  path=data_path)

    # Save data frames on file
    path = "../../resources/boosting_dataframe/"
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except FileNotFoundError as e:
        os.makedirs(path)

    train_df.to_csv(path + "train_df_{}_advanced_lt_{}.csv".format(TRAIN_CUTOFF, LOWER_THRESHOLD), index=False)
    valid_df.to_csv(path + "valid_df_{}_advanced_lt_{}.csv".format(VALID_CUTOFF, LOWER_THRESHOLD), index=False)
