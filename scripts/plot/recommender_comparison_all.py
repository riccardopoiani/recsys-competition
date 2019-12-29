from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_UCM_train, get_ICM_train_new
from src.data_management.dataframe_preprocessing import get_preprocessed_dataframe
from src.feature.demographics_content import get_user_demographic
from src.model import new_best_models, k_1_out_best_models
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
from src.plots.recommender_plots import *
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    root_data_path = "../../data"
    k_out = 1
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=k_out, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    ICM_all, _ = get_ICM_train_new(data_reader)
    UCM_all = get_UCM_train(data_reader)

    item_cbf_cf_parm = new_best_models.ItemCBF_CF.get_best_parameters()
    item_cbf_cf_parm['topK'] = 5
    model = ItemKNNCBFCFRecommender(URM_train=URM_train, ICM_train=ICM_all)
    model.fit(**item_cbf_cf_parm)
    model.RECOMMENDER_NAME = "ItemCBFCF"

    sub4 = k_1_out_best_models.HybridNormWeightedAvgAll.get_model(URM_train, ICM_all, UCM_all)
    recommender_list = [model, sub4]

    # Building path
    version_path = "../../report/graphics/comparison/"

    # Plotting the comparison on age
    UCM_region = data_reader.get_UCM_from_name("UCM_region")
    region_feature_to_id_mapper = data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_region")
    region_demographic = get_user_demographic(UCM_region, region_feature_to_id_mapper, binned=True)
    region_demographic_describer_list = [-1, 0, 2, 3, 4, 5, 6, 7]
    demographic_plot(recommender_instance_list=recommender_list, URM_train=URM_train,
                     URM_test=URM_test, cutoff=10, metric="MAP", save_on_file=True,
                     output_folder_path=version_path + "region/", demographic_name="Region",
                     demographic=region_demographic, demographic_describer_list=region_demographic_describer_list,
                     exclude_cold_users=True, foh=-1)

    # Plotting the comparison on region
    UCM_age = data_reader.get_UCM_from_name("UCM_age")
    age_feature_to_id_mapper = data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_age")
    age_demographic = get_user_demographic(UCM_age, age_feature_to_id_mapper, binned=True)
    age_demographic_describer_list = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    demographic_plot(recommender_instance_list=recommender_list, URM_train=URM_train,
                     URM_test=URM_test, cutoff=10, metric="MAP", save_on_file=True,
                     output_folder_path=version_path + "age/", demographic_name="Age",
                     demographic=age_demographic, demographic_describer_list=age_demographic_describer_list,
                     exclude_cold_users=True, foh=-1)

    # Plot on profile length
    plot_compare_recommenders_user_profile_len(recommender_list, URM_train, URM_test, save_on_file=True,
                                               output_folder_path=version_path + "user_activity/",
                                               bins=30)

    # Plotting the comparison based on clustering
    dataframe = get_preprocessed_dataframe(path="../../data/", keep_warm_only=True)
    plot_clustering_demographics(recommender_list, URM_train, URM_test, dataframe,
                                 metric="MAP", cutoff=10, save_on_file=True,
                                 output_folder_path=version_path + "clustering/",
                                 exclude_cold_users=True, n_clusters=10,
                                 n_init=10)
