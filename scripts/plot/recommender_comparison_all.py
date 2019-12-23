from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import merge_UCM, get_ICM_numerical
from src.data_management.data_reader import get_UCM_train, get_ICM_train_new
from src.feature.demographics_content import get_user_demographic
from src.model import new_best_models
from src.model.Ensemble.Boosting.boosting_preprocessing import preprocess_dataframe_after_reading, add_label
from src.plots.recommender_plots import *
from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    root_data_path = "../../data"
    k_out = 3
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=k_out, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()

    ICM_all_new, _ = get_ICM_train_new(data_reader)
    UCM_all_new = get_UCM_train(data_reader)
    ICM_numerical, _ = get_ICM_numerical(data_reader.dataReader_object)
    ICM = data_reader.get_ICM_from_name("ICM_all")
    ICM_subclass = data_reader.get_ICM_from_name("ICM_sub_class")

    # Build UCMs
    URM_all = data_reader.dataReader_object.get_URM_all()
    UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
    UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
    UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})

    # Read boosting data
    dataframe_path = "../../boosting_dataframe/"
    train_df = pd.read_csv(dataframe_path + "train_df_20_advanced_foh_5.csv")
    valid_df = pd.read_csv(dataframe_path + "valid_df_20_advanced_foh_5.csv")

    train_df = preprocess_dataframe_after_reading(train_df)
    train_df = train_df.drop(columns=["label"], inplace=False)
    valid_df = preprocess_dataframe_after_reading(valid_df)

    y_train, non_zero_count, total = add_label(data_frame=train_df, URM_train=URM_train)

    boosting = new_best_models.BoostingFoh5.get_model(URM_train=URM_train, train_df=train_df, y_train=y_train,
                                                      valid_df=valid_df,
                                                      model_path="../../report/hp_tuning/boosting/Dec23_11-22"
                                                                 "-35_k_out_value_3_eval/best_model6")
    boosting.RECOMMENDER_NAME = "Boosting"
    mixed_item = new_best_models.MixedItem.get_model(URM_train, ICM_all_new)
    mixed_item.RECOMMENDER_NAME = "MixedItem"

    recommender_list = [boosting, mixed_item]

    # Building path
    version_path = "../../report/graphics/comparison/"

    # TODO FIX get_user_demographic --> need to do the mapping of the features to the original values
    # Plotting the comparison on age
    region_demographic = get_user_demographic(UCM_region, URM_train, k_out, binned=True)
    region_demographic_describer_list = [-1, 0, 2, 3, 4, 5, 6, 7]
    demographic_plot(recommender_instance_list=recommender_list, URM_train=URM_train,
                     URM_test=URM_test, cutoff=10, metric="MAP", save_on_file=True,
                     output_folder_path=version_path + "region/", demographic_name="Region",
                     demographic=region_demographic, demographic_describer_list=region_demographic_describer_list,
                     exclude_cold_users=True, foh=5)

    # Plotting the comparison on region
    age_demographic = get_user_demographic(UCM_age, URM_all, k_out, binned=True)
    age_demographic_describer_list = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    demographic_plot(recommender_instance_list=recommender_list, URM_train=URM_train,
                     URM_test=URM_test, cutoff=10, metric="MAP", save_on_file=True,
                     output_folder_path=version_path + "age/", demographic_name="Age",
                     demographic=age_demographic, demographic_describer_list=age_demographic_describer_list,
                     exclude_cold_users=True, foh=5)

    # Plot on profile length
    # plot_compare_recommenders_user_profile_len(recommender_list, URM_train, URM_test, save_on_file=True,
    #                                           output_folder_path=version_path + "user_activity/",
    #                                           bins=30)

    # Plotting the comparison based on clustering
    # dataframe = get_preprocessed_dataframe(path="../../data/", keep_warm_only=True)
    # plot_clustering_demographics(recommender_list, URM_train, URM_test, dataframe,
    #                             metric="MAP", cutoff=10, save_on_file=True,
    #                             output_folder_path=version_path + "clustering/",
    #                             exclude_cold_users=True, n_clusters=80,
    #                             n_init=10)
