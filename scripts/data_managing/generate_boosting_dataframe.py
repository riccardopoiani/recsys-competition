import os

from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model import new_best_models
from src.model.Ensemble.Boosting.boosting_preprocessing import get_train_dataframe_proportion, \
    get_valid_dataframe_second_version
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    # Data loading
    root_data_path = "../../data/"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    # Build ICMs
    ICM_all = get_ICM_train(data_reader)

    # Build UCMs: do not change the order of ICMs and UCMs
    UCM_all = get_UCM_train(data_reader)

    exclude_users_mask = np.ediff1d(URM_train.tocsr().indptr) < 5
    exclude_users = np.arange(URM_train.shape[0])[exclude_users_mask]

    # XGB SETUP
    main_rec = new_best_models.MixedItem.get_model(URM_train=URM_train, ICM_all=ICM_all)
    sub_0 = main_rec
    sub_0.RECOMMENDER_NAME = "MixedItem"
    sub_1 = new_best_models.ItemCBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all)
    sub_1.RECOMMENDER_NAME = "ItemCBF_CF"
    sub_2 = new_best_models.RP3BetaSideInfo.get_model(URM_train=URM_train, ICM_train=ICM_all)
    sub_2.RECOMMENDER_NAME = "RP3BetaSideInfo"
    sub_3 = new_best_models.UserCBF_CF_Warm.get_model(URM_train=URM_train, UCM_train=UCM_all)
    sub_3.RECOMMENDER_NAME = "UserCBF_CF"
    sub_4 = new_best_models.ItemCBF_all_FW.get_model(URM_train=URM_train, ICM_train=ICM_all)
    sub_4.RECOMMENDER_NAME = "ItemCBF_all_FW"
    sub_5 = new_best_models.UserCF.get_model(URM_train=URM_train)
    sub_5.RECOMMENDER_NAME = "UserCF"

    sub_list = [sub_0, sub_1, sub_2, sub_3, sub_4, sub_5]

    # Retrieve data for boosting
    mapper = data_reader.SPLIT_GLOBAL_MAPPER_DICT['user_original_ID_to_index']

    main_recommender = main_rec
    total_users = np.arange(URM_train.shape[0])
    mask = np.in1d(total_users, exclude_users, invert=True)
    user_to_validate = total_users[mask]
    cutoff = 20
    data_path = "../../data/"

    train_df = get_train_dataframe_proportion(user_id_array=user_to_validate,
                                              cutoff=cutoff,
                                              main_recommender=main_recommender,
                                              recommender_list=sub_list,
                                              mapper=mapper, URM_train=URM_train, path=data_path,
                                              proportion=0.5)

    valid_df = get_valid_dataframe_second_version(user_id_array=user_to_validate,
                                                  cutoff=cutoff,
                                                  main_recommender=main_recommender,
                                                  recommender_list=sub_list,
                                                  mapper=mapper,
                                                  URM_train=URM_train,
                                                  path=data_path)

    # Save data frames on file
    path = "../../boosting_dataframe/"
    try:
        if not os.path.exists(path):
            os.mkdir(path)
    except FileNotFoundError as e:
        os.makedirs(path)

    train_df.to_csv(path + "train_df_{}_advanced_foh_5.csv".format(cutoff), index=False)
    valid_df.to_csv(path + "valid_df_{}_advanced_foh_5.csv".format(cutoff), index=False)
